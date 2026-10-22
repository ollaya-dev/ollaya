// Static build: everything the site needs ends up in dist/ as plain files.
//
//   1. A fresh output directory gets a copy of public/ (favicon, _headers, _redirects, static/*).
//   2. Tailwind compiles src/styles/app.css → dist/static/app.css.
//   3. Files in dist/static are hashed (for ?v=<hash> cache busting).
//   4. src/build.tsx is bundled with esbuild and renderSite() pre-renders every page, 404.html,
//      robots.txt, sitemap.xml and search.json.
//   5. /install.sh is a copy of the real installer, ../scripts/install.sh.
//   6. The static registry (../registry/{v2,blobs}) is copied in; the result is synced into dist/.
//
// SITE_ORIGIN (env) sets the public origin used in canonical/OG URLs, the install command,
// robots.txt and sitemap.xml. Run `npm run gen` first (npm run build does).

import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { cp, mkdir, readdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { dirname, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { build } from 'esbuild'

const DEFAULT_ORIGIN = 'https://ollaya.dev'
const DEFAULT_BENCH_ORIGIN = 'https://bench.ollaya.dev'

const root = fileURLToPath(new URL('..', import.meta.url))
const finalDist = join(root, 'dist')
// Build into a sibling directory first, then sync it into dist/ (see the end of this file).
const dist = join(root, '.dist-tmp')
const cacheDir = join(root, 'node_modules', '.cache', 'ollaya-site')

function resolveOrigin(value) {
  try {
    const url = new URL(value)
    if (url.protocol !== 'https:' && url.protocol !== 'http:') throw new Error('not http(s)')
    return url.origin
  } catch {
    throw new Error(`SITE_ORIGIN must be an absolute http(s) URL, got "${value}"`)
  }
}

// The repository's star count for the header, read from GitHub's API at build time (the site has no
// client-side requests to other origins). An offline or rate-limited build shows the link without it.
async function githubStars() {
  try {
    const res = await fetch('https://api.github.com/repos/ollaya-dev/ollaya', {
      headers: {
        Accept: 'application/vnd.github+json',
        ...(process.env.GITHUB_TOKEN ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` } : {}),
      },
      signal: AbortSignal.timeout(5000),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const n = (await res.json()).stargazers_count
    if (!Number.isInteger(n)) throw new Error('no stargazers_count')
    return n
  } catch (e) {
    console.warn(`build: GitHub star count unavailable (${e.message}); the header shows none`)
    return undefined
  }
}

async function listFiles(dir) {
  const out = []
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name)
    if (e.isDirectory()) out.push(...(await listFiles(p)))
    else out.push(p)
  }
  return out
}

const origin = resolveOrigin(process.env.SITE_ORIGIN?.trim() || DEFAULT_ORIGIN)
const started = Date.now()

for (const generated of ['content.ts', 'registry.ts']) {
  try {
    await stat(join(root, 'src', 'generated', generated))
  } catch {
    throw new Error(`src/generated/${generated} is missing — run \`npm run gen\` first`)
  }
}

// 1. Fresh dist/ with the public files.
await rm(dist, { recursive: true, force: true })
await cp(join(root, 'public'), dist, { recursive: true })

// 2. Tailwind.
const tailwind = spawnSync(
  join(root, 'node_modules', '.bin', 'tailwindcss'),
  ['-i', 'src/styles/app.css', '-o', join(dist, 'static', 'app.css'), '--minify'],
  { cwd: root, stdio: ['ignore', 'ignore', 'pipe'], encoding: 'utf8' },
)
if (tailwind.status !== 0) {
  process.stderr.write(tailwind.stderr ?? '')
  throw new Error('tailwindcss failed')
}

// 2b. The /try demo runs ../web/ollaya.js in the browser. Its page script and two runtime bundles (web/ollaya.js
// with onnxruntime-web's WebAssembly build, and with its JSPI build for WebGPU; the other WebGPU builds' .wasm is
// over Cloudflare's 25 MiB file limit) go to dist/static with the .wasm and .mjs files they load, so no CDN is
// needed. web/pkg is built from Rust (sh ../web/build.sh) and CI does not build it: without it /try is left out.
const web = join(root, '..', 'web')
const ortDist = join(web, 'node_modules', 'onnxruntime-web', 'dist')
const ortFiles = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.jspi.wasm', 'ort-wasm-simd-threaded.jspi.mjs']
if ((await stat(join(web, 'pkg', 'ollaya_web_bg.wasm')).catch(() => null)) && (await stat(ortDist).catch(() => null))) {
  const browser = { bundle: true, format: 'esm', platform: 'browser', target: 'es2022', minify: true, logLevel: 'warning' }
  await build({ ...browser, entryPoints: [join(root, 'src', 'client', 'try.js')], outfile: join(dist, 'static', 'try.js') })
  for (const [variant, ortBuild] of [['wasm', 'onnxruntime-web/wasm'], ['webgpu', 'onnxruntime-web/jspi']]) {
    await build({
      ...browser,
      entryPoints: [join(web, 'ollaya.js')],
      outfile: join(dist, 'static', `ollaya-${variant}.js`),
      alias: { 'onnxruntime-web': ortBuild },
      // An alias resolves from the working directory: onnxruntime-web is web/'s dependency, and this condition picks
      // the build that loads its .mjs and .wasm from env.wasm.wasmPaths rather than inlining them.
      nodePaths: [join(web, 'node_modules')],
      conditions: ['onnxruntime-web-use-extern-wasm'],
    })
  }
  await cp(join(web, 'pkg', 'ollaya_web_bg.wasm'), join(dist, 'static', 'ollaya_web_bg.wasm'))
  for (const f of ortFiles) await cp(join(ortDist, f), join(dist, 'static', f))
} else {
  console.warn('build: ../web/pkg is not built (sh ../web/build.sh, then npm ci in ../web); /try is left out')
}

// 3. Content hashes of static files.
const staticDir = join(dist, 'static')
const assetVersions = {}
for (const file of await listFiles(staticDir)) {
  const hash = createHash('sha256').update(await readFile(file)).digest('hex').slice(0, 10)
  assetVersions[relative(staticDir, file).split(sep).join('/')] = hash
}

// 4. Pre-render.
const bundle = join(cacheDir, 'build.mjs')
await build({
  entryPoints: [join(root, 'src', 'build.tsx')],
  outfile: bundle,
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node22',
  jsx: 'automatic',
  jsxImportSource: 'hono/jsx',
  logLevel: 'warning',
})
const { renderSite, withReadmeBlocks } = await import(`${pathToFileURL(bundle).href}?t=${Date.now()}`)
// The README's measured numbers come from results/ (src/data/readme.ts): README=write rewrites them, otherwise a stale
// block fails the build.
const readmePath = join(root, '..', 'README.md')
const readme = await readFile(readmePath, 'utf8')
// GitHub shows the README, so its links go to the production site whatever SITE_ORIGIN this build uses.
const readmeNow = withReadmeBlocks(readme, DEFAULT_ORIGIN)
if (readmeNow !== readme) {
  if (process.env.README !== 'write') throw new Error('README.md: its results blocks are stale; run `npm run readme` in site/')
  await writeFile(readmePath, readmeNow)
  console.log('README.md: results blocks rewritten')
}
const bench = resolveOrigin(process.env.BENCH_ORIGIN?.trim() || DEFAULT_BENCH_ORIGIN)
const files = await renderSite({ origin, bench, assetVersions, stars: await githubStars() })

for (const f of files) {
  const target = join(dist, f.path)
  if (relative(dist, target).startsWith('..')) throw new Error(`refusing to write outside dist/: ${f.path}`)
  await mkdir(dirname(target), { recursive: true })
  await writeFile(target, f.body)
}

// The real installer: `curl -fsSL <origin>/install.sh | sh` serves ../scripts/install.sh unchanged.
const installer = join(root, '..', 'scripts', 'install.sh')
const installText = await readFile(installer, 'utf8').catch(() => {
  throw new Error(`${installer} is missing: /install.sh must serve the real installer`)
})
if (!installText.startsWith('#!/bin/sh')) throw new Error(`${installer} does not start with #!/bin/sh`)
await writeFile(join(dist, 'install.sh'), installText)
// The Windows CLI installer: `irm <origin>/install.ps1 | iex` serves ../scripts/install.ps1 unchanged.
await writeFile(join(dist, 'install.ps1'), await readFile(join(root, '..', 'scripts', 'install.ps1')))

// The static model registry (../registry, written by convert/ollaya_convert/package.py): manifests
// under /v2/ and derived blobs under /blobs/. Manifests carry absolute blob URLs, so they must
// have been packaged for the origin this site is built for. The manifests include every one ever
// published under a tag, kept by digest (manifests/sha256-<hex>; docs/decisions/0006-manifest-retention.md),
// so their blobs must all be here too: a deploy never drops what an older manifest needs.
const registry = process.env.OLLAYA_REGISTRY_DIR ? resolve(process.env.OLLAYA_REGISTRY_DIR) : join(root, '..', 'registry')
let registryFiles = 0
if (await stat(registry).catch(() => null)) {
  for (const sub of ['v2', 'blobs']) {
    const src = join(registry, sub)
    if (await stat(src).catch(() => null)) await cp(src, join(dist, sub), { recursive: true })
  }
  const walk = async (d) => (await readdir(d, { withFileTypes: true })).flatMap((e) => (e.isDirectory() ? [] : [join(d, e.name)]))
  const manifestDirs = []
  const collect = async (d) => {
    for (const e of await readdir(d, { withFileTypes: true })) {
      if (e.isDirectory()) await collect(join(d, e.name))
      else if (d.endsWith('/manifests')) manifestDirs.push(join(d, e.name))
    }
  }
  if (await stat(join(dist, 'v2')).catch(() => null)) await collect(join(dist, 'v2'))
  // Every blob a manifest points at on this origin must be in the build, byte for byte: a deploy
  // from a checkout without ../registry/blobs (gitignored) would otherwise break every pull.
  const missing = new Set()
  const checked = new Set()
  for (const m of manifestDirs) {
    const manifest = JSON.parse(await readFile(m, 'utf8'))
    for (const layer of [manifest.config, ...manifest.layers]) {
      for (const url of layer.urls ?? []) {
        if (!url.includes('/blobs/sha256-')) continue
        if (!url.startsWith(`${origin}/blobs/`)) {
          throw new Error(`${relative(dist, m)}: blob URL ${url} is not under ${origin}; rerun package.py with SITE_ORIGIN=${origin}`)
        }
        const name = url.slice(`${origin}/blobs/`.length)
        if (checked.has(name)) continue
        checked.add(name)
        const body = await readFile(join(dist, 'blobs', name)).catch(() => null)
        if (!body) {
          missing.add(name)
          continue
        }
        const digest = `sha256:${createHash('sha256').update(body).digest('hex')}`
        if (digest !== layer.digest) throw new Error(`registry/blobs/${name} is ${digest}, but ${relative(dist, m)} expects ${layer.digest}`)
      }
    }
  }
  if (missing.size) {
    const message =
      `${missing.size} blob(s) the manifests point to are not in ../registry/blobs (derived files are not in git), ` +
      `e.g. ${[...missing][0]}. Download the published ones with: node scripts/fetch-blobs.mjs`
    // CI builds the pages to check them and never deploys, so it may go without the blobs.
    if (process.env.OLLAYA_ALLOW_MISSING_BLOBS !== '1') throw new Error(message)
    console.warn(`warning: ${message}`)
  }
  registryFiles = manifestDirs.length + (await walk(join(dist, 'blobs')).catch(() => [])).length
}

// Sync the fresh build into dist/ file by file: a running `wrangler dev` keeps watching the same
// directory (replacing the whole directory at once can leave it serving 500s).
async function removeEmptyDirs(dir) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    if (!e.isDirectory()) continue
    const p = join(dir, e.name)
    await removeEmptyDirs(p)
    if ((await readdir(p)).length === 0) await rm(p, { recursive: true })
  }
}
const fresh = new Set((await listFiles(dist)).map((f) => relative(dist, f)))
const existing = (await listFiles(finalDist).catch(() => [])).map((f) => relative(finalDist, f))
for (const f of existing) if (!fresh.has(f)) await rm(join(finalDist, f), { force: true })
for (const f of fresh) {
  const body = await readFile(join(dist, f))
  const target = join(finalDist, f)
  const current = await readFile(target).catch(() => null)
  if (current && current.equals(body)) continue
  await mkdir(dirname(target), { recursive: true })
  await writeFile(`${target}.tmp`, body)
  await rename(`${target}.tmp`, target)
}
await removeEmptyDirs(finalDist)
await rm(dist, { recursive: true, force: true })
if (registryFiles) console.log(`registry: ${registryFiles} manifests and blobs copied into dist/`)

const pagesCount = files.filter((f) => f.path.endsWith('.html')).length
console.log(
  `build: ${pagesCount} HTML pages + ${files.length - pagesCount} other files → dist/ for ${origin} (${Date.now() - started} ms)`,
)
