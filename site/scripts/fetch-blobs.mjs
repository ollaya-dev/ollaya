// Downloads the derived registry blobs (ONNX graphs, configs, calibration) that the manifests in
// ../registry/v2 point to, into ../registry/blobs, checking each one's sha256. Those files are
// build outputs of convert/ollaya_convert/package.py and are not in git, so a fresh checkout needs
// this before `npm run build` or a deploy. Blobs already present are kept; weights are never
// fetched (they stay on their authors' Hugging Face repos).
//
//   node scripts/fetch-blobs.mjs

import { createHash } from 'node:crypto'
import { mkdir, readdir, readFile, rename, stat, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const registry = process.env.OLLAYA_REGISTRY_DIR || fileURLToPath(new URL('../../registry', import.meta.url))
const blobs = join(registry, 'blobs')

async function manifests(dir) {
  const out = []
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name)
    if (e.isDirectory()) out.push(...(await manifests(p)))
    else if (dir.endsWith('/manifests')) out.push(p)
  }
  return out
}

const wanted = new Map()
for (const m of await manifests(join(registry, 'v2'))) {
  const manifest = JSON.parse(await readFile(m, 'utf8'))
  for (const layer of [manifest.config, ...manifest.layers]) {
    for (const url of layer.urls ?? []) {
      const at = url.indexOf('/blobs/sha256-')
      if (at >= 0) wanted.set(url.slice(at + '/blobs/'.length), { url, digest: layer.digest })
    }
  }
}

await mkdir(blobs, { recursive: true })
let fetched = 0
for (const [name, { url, digest }] of wanted) {
  const path = join(blobs, name)
  if (await stat(path).catch(() => null)) continue
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`)
  const body = Buffer.from(await res.arrayBuffer())
  const got = `sha256:${createHash('sha256').update(body).digest('hex')}`
  if (got !== digest) throw new Error(`${url}: expected ${digest}, got ${got}`)
  await writeFile(`${path}.tmp`, body)
  await rename(`${path}.tmp`, path)
  fetched++
}
console.log(`blobs: ${wanted.size} referenced, ${fetched} downloaded into ${blobs}`)
