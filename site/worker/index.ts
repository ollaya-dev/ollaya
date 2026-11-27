// The site's Worker. Every page and file is a static asset; this script runs only for /api/* (run_worker_first in
// wrangler.jsonc) and for paths that match no asset, which it hands back to the assets (dist/404.html).
//
// /api/cloud/* runs Ollaya Cloud requests for a signed-in visitor. A browser cannot call api.ollaya.dev itself (the
// gateway refuses any request with an Origin header) and an API key must never reach the browser, so:
//   1. the browser gets a 5-minute JWT from /dashboard/api/auth/token (Ollaya Cloud's sign-in service) and sends it
//      here as a Bearer token;
//   2. this Worker checks the JWT against the sign-in service's JWKS (service binding AUTH);
//   3. it finds the visitor's inference key for the organization in KV (CLOUD_KEYS, AES-GCM with the KEYS_SECRET
//      secret), or makes one through the gateway with the JWT (service binding GATEWAY);
//   4. it sends the request to the gateway with the key and returns the answer. On a 401 it makes a new key once.
// Each request is billed to the visitor's own organization. Never fetch ollaya.dev/dashboard/* with plain fetch():
// a Worker's request to its own zone does not reach the route Workers, so the service bindings are the way in.

import { createLocalJWKSet, jwtVerify, type JSONWebKeySet } from 'jose'

interface Fetcher {
  fetch(input: string | Request, init?: RequestInit): Promise<Response>
}

interface KVNamespace {
  get(key: string): Promise<string | null>
  put(key: string, value: string, options?: { expirationTtl?: number }): Promise<void>
  delete(key: string): Promise<void>
}

export interface Env {
  ASSETS: Fetcher
  AUTH: Fetcher
  GATEWAY: Fetcher
  CLOUD_KEYS: KVNamespace
  /** 32 random bytes, base64: `openssl rand -base64 32 | wrangler secret put KEYS_SECRET`. */
  KEYS_SECRET: string
}

const ISSUER = 'https://ollaya.dev'
const AUDIENCE = 'control-plane'
const JWKS_URL = 'https://ollaya.dev/dashboard/api/auth/jwks'
const API = 'https://api.ollaya.dev'
const JWKS_TTL_MS = 10 * 60 * 1000
const CATALOG_TTL_MS = 10 * 60 * 1000
/** The gateway's keys last 30 days; KV forgets them a day earlier, so a stored key is never an expired one. */
const KEY_DAYS = 30
const KV_TTL_S = (KEY_DAYS - 1) * 24 * 60 * 60
const MAX_BODY = 256 * 1024
const ID = /^[A-Za-z0-9_-]{1,64}$/

class Failure extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message)
  }
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' },
  })

const fail = (status: number, code: string, error: string) => json({ error, code }, status)

// --- The visitor -----------------------------------------------------------------------------------------

let jwks: { set: ReturnType<typeof createLocalJWKSet>; at: number } | undefined

async function keySet(env: Env, fresh = false) {
  if (!fresh && jwks && Date.now() - jwks.at < JWKS_TTL_MS) return jwks.set
  const r = await env.AUTH.fetch(JWKS_URL, { headers: { accept: 'application/json' } })
  if (!r.ok) throw new Failure(502, 'JWKS_UNAVAILABLE', `the sign-in service answered ${r.status} for its keys`)
  jwks = { set: createLocalJWKSet((await r.json()) as JSONWebKeySet), at: Date.now() }
  return jwks.set
}

/** The signed-in visitor from the request's Bearer JWT: the user id and the JWT itself. */
async function visitor(request: Request, env: Env) {
  const token = /^Bearer (\S+)$/.exec(request.headers.get('authorization') ?? '')?.[1]
  if (!token) throw new Failure(401, 'SIGN_IN', 'sign in first: send Authorization: Bearer <JWT from /dashboard/api/auth/token>')
  const verify = async (fresh: boolean) =>
    jwtVerify(token, await keySet(env, fresh), { issuer: ISSUER, audience: AUDIENCE, algorithms: ['EdDSA'] })
  let payload
  try {
    payload = (await verify(false)).payload
  } catch (e) {
    // A key the cached set does not have yet: the sign-in service rotated its keys.
    if ((e as { code?: string }).code !== 'ERR_JWKS_NO_MATCHING_KEY') throw new Failure(401, 'SIGN_IN', 'the sign-in token is not valid or has expired')
    try {
      payload = (await verify(true)).payload
    } catch {
      throw new Failure(401, 'SIGN_IN', 'the sign-in token is not valid or has expired')
    }
  }
  const id = payload.id
  if (typeof id !== 'string' || !ID.test(id)) throw new Failure(401, 'SIGN_IN', 'the sign-in token has no user id')
  return { id, token }
}

// --- The visitor's inference key, encrypted in KV ---------------------------------------------------------

const b64 = (bytes: Uint8Array) => btoa(String.fromCharCode(...bytes))
const unb64 = (s: string) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0))

let aes: { secret: string; key: Promise<CryptoKey> } | undefined

function aesKey(env: Env) {
  if (aes?.secret !== env.KEYS_SECRET) {
    const raw = unb64(env.KEYS_SECRET ?? '')
    if (raw.length !== 32) throw new Failure(500, 'NOT_CONFIGURED', 'the site has no KEYS_SECRET of 32 bytes')
    aes = { secret: env.KEYS_SECRET, key: crypto.subtle.importKey('raw', raw, 'AES-GCM', false, ['encrypt', 'decrypt']) }
  }
  return aes.key
}

// The KV name goes in as additional data, so a value copied to another visitor's name does not decrypt.
async function seal(env: Env, name: string, plain: string) {
  const iv = crypto.getRandomValues(new Uint8Array(12))
  const data = new TextEncoder().encode(name)
  const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv, additionalData: data }, await aesKey(env), new TextEncoder().encode(plain))
  return `v1.${b64(iv)}.${b64(new Uint8Array(ct))}`
}

async function open(env: Env, name: string, sealed: string) {
  const [v, iv, ct] = sealed.split('.')
  if (v !== 'v1' || !iv || !ct) return null
  try {
    const data = new TextEncoder().encode(name)
    const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: unb64(iv), additionalData: data }, await aesKey(env), unb64(ct))
    return new TextDecoder().decode(plain)
  } catch {
    return null
  }
}

/** A gateway request: never with an Origin header, which the gateway refuses. */
function gateway(env: Env, path: string, auth: string, body?: unknown) {
  return env.GATEWAY.fetch(`${API}${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { authorization: `Bearer ${auth}`, accept: 'application/json', ...(body === undefined ? {} : { 'content-type': 'application/json' }) },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
}

async function newKey(env: Env, name: string, jwt: string, org: string) {
  const r = await gateway(env, `/v1/orgs/${org}/keys`, jwt, { name: 'ollaya.dev web', scope: 'inference', expires_in_days: KEY_DAYS })
  if (!r.ok) {
    const answer = (await r.json().catch(() => null)) as { error?: string; code?: string } | null
    throw new Failure(r.status === 401 || r.status === 403 || r.status === 404 ? r.status : 502, answer?.code ?? 'KEY_FAILED', answer?.error ?? `the gateway answered ${r.status} for a new key`)
  }
  const { key } = (await r.json()) as { key?: unknown }
  if (typeof key !== 'string' || !key) throw new Failure(502, 'KEY_FAILED', 'the gateway gave no key')
  await env.CLOUD_KEYS.put(name, await seal(env, name, key), { expirationTtl: KV_TTL_S })
  return key
}

async function storedKey(env: Env, name: string) {
  const sealed = await env.CLOUD_KEYS.get(name)
  return sealed ? open(env, name, sealed) : null
}

// --- Routes ----------------------------------------------------------------------------------------------

async function decide(request: Request, env: Env) {
  if (request.method !== 'POST') return fail(405, 'METHOD', 'use POST')
  const who = await visitor(request, env)
  const text = await request.text()
  if (text.length > MAX_BODY) return fail(413, 'TOO_LARGE', `the request is over ${MAX_BODY / 1024} KB`)
  let body: Record<string, unknown>
  try {
    body = JSON.parse(text)
  } catch {
    return fail(400, 'BAD_JSON', 'the body is not JSON')
  }
  const { org, ...req } = body ?? {}
  if (typeof org !== 'string' || !ID.test(org)) return fail(400, 'BAD_ORG', 'the body needs "org": the id of one of your organizations')
  if (typeof req.model !== 'string' || !req.questions || typeof req.questions !== 'object') {
    return fail(400, 'BAD_REQUEST', 'the body needs "model" and a "questions" object, as /api/decide')
  }

  const name = `key:${who.id}:${org}`
  let key = (await storedKey(env, name)) ?? (await newKey(env, name, who.token, org))
  let r = await gateway(env, '/api/decide', key, req)
  if (r.status === 401) {
    // The key was revoked or has expired: make a new one and try once more.
    await env.CLOUD_KEYS.delete(name)
    key = await newKey(env, name, who.token, org)
    r = await gateway(env, '/api/decide', key, req)
  }
  return new Response(r.body, {
    status: r.status,
    headers: { 'content-type': r.headers.get('content-type') ?? 'application/json', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' },
  })
}

let catalog: { body: string; at: number } | undefined

/** Ollaya Cloud's model list (public), for the browser, which cannot call the gateway itself. */
async function models(env: Env) {
  if (!catalog || Date.now() - catalog.at > CATALOG_TTL_MS) {
    const r = await env.GATEWAY.fetch(`${API}/v1/catalog`, { headers: { accept: 'application/json' } })
    if (!r.ok) return fail(502, 'CATALOG_UNAVAILABLE', `the gateway answered ${r.status} for its catalog`)
    catalog = { body: await r.text(), at: Date.now() }
  }
  return new Response(catalog.body, {
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'public, max-age=300', 'x-content-type-options': 'nosniff' },
  })
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { pathname } = new URL(request.url)
    if (!pathname.startsWith('/api/')) return env.ASSETS.fetch(request)
    try {
      if (pathname === '/api/cloud/decide') return await decide(request, env)
      if (pathname === '/api/cloud/models' && request.method === 'GET') return await models(env)
      return fail(404, 'NOT_FOUND', 'no such API route')
    } catch (e) {
      if (e instanceof Failure) return fail(e.status, e.code, e.message)
      console.error('api error', pathname, e)
      return fail(500, 'INTERNAL', 'the site could not run the request')
    }
  },
}
