// Static build: everything the site needs ends up in dist/ as plain files.
//
//   1. A fresh output directory gets a copy of public/ (favicon, _headers, _redirects, static/*).
//   2. Tailwind compiles src/styles/app.css → dist/static/app.css.
//   3. Files in dist/static are hashed (for ?v=<hash> cache busting).
//   4. src/build.tsx is bundled with esbuild and renderSite() pre-renders every page, 404.html,
//      install.sh, robots.txt, sitemap.xml and search.json; the result replaces dist/.
//
// SITE_ORIGIN (env) sets the public origin used in canonical/OG URLs, the install command,
// robots.txt and sitemap.xml. Run `node scripts/gen-content.mjs` first (npm run build does).

import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { cp, mkdir, readdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { dirname, join, relative, sep } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { build } from 'esbuild'

const DEFAULT_ORIGIN = 'https://ollaya.cobanov.dev'

const root = fileURLToPath(new URL('..', import.meta.url))
const finalDist = join(root, 'dist')
// Build into a sibling directory and swap it in at the end, so a running `wrangler dev`
// never sees a half-written dist/.
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

try {
  await stat(join(root, 'src', 'generated', 'content.ts'))
} catch {
  throw new Error('src/generated/content.ts is missing — run `node scripts/gen-content.mjs` first')
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
const { renderSite } = await import(`${pathToFileURL(bundle).href}?t=${Date.now()}`)
const files = await renderSite({ origin, assetVersions })

for (const f of files) {
  const target = join(dist, f.path)
  if (relative(dist, target).startsWith('..')) throw new Error(`refusing to write outside dist/: ${f.path}`)
  await mkdir(dirname(target), { recursive: true })
  await writeFile(target, f.body)
}

await rm(finalDist, { recursive: true, force: true })
await rename(dist, finalDist)

const pagesCount = files.filter((f) => f.path.endsWith('.html')).length
console.log(
  `build: ${pagesCount} HTML pages + ${files.length - pagesCount} other files → dist/ for ${origin} (${Date.now() - started} ms)`,
)
