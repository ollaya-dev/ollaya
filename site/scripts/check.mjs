// Post-build checks on dist/ (run by `npm run build`). Exits non-zero if anything fails.
//
//  1. Required files exist; the site itself puts nothing under the reserved /v2/ prefix.
//  2. The stylesheet contains the design tokens.
//  3. No public host name is hard-coded in sources (it must come from SITE_ORIGIN).
//  4. Every internal href/src in every page resolves to a file in dist/.
//  5. /install.sh is the real installer (../scripts/install.sh), byte for byte.
//  6. Every tag's manifest under /v2/ is also there by its digest (sha256-<hex>, served as
//     sha256:<hex> through _redirects), and every manifest named by a digest hashes to it.

import { createHash } from 'node:crypto'
import { readdir, readFile, stat } from 'node:fs/promises'
import { join, relative, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const dist = join(root, 'dist')
const errors = []

async function walk(dir) {
  const out = []
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name)
    if (e.isDirectory()) out.push(...(await walk(p)))
    else out.push(p)
  }
  return out
}
const exists = (p) => stat(p).then(() => true, () => false)

// 1. Required files ----------------------------------------------------------------------------
const required = [
  'index.html',
  '404.html',
  'search.html',
  'search.json',
  'download.html',
  'docs.html',
  'library.html',
  'install.sh',
  'robots.txt',
  'sitemap.xml',
  'favicon.svg',
  '_headers',
  '_redirects',
  'static/app.css',
  'static/app.js',
]
for (const f of required) if (!(await exists(join(dist, f)))) errors.push(`dist/${f} is missing`)
// /v2/ is reserved for registry manifests, which the packaging step copies into dist/ later.
if (await exists(join(root, 'public', 'v2'))) errors.push('public/v2 exists — /v2/ is reserved for the model registry')

const files = (await walk(dist)).map((f) => relative(dist, f).split(sep).join('/'))
const fileSet = new Set(files)

// 2. CSS ---------------------------------------------------------------------------------------
const css = await readFile(join(dist, 'static/app.css'), 'utf8').catch(() => '')
for (const token of ['--color-canvas', '--color-fg', '.frame', '.prose']) {
  if (!css.includes(token)) errors.push(`dist/static/app.css is missing ${token}`)
}

// 3. Hard-coded hosts in sources -------------------------------------------------------------------
const hostPattern = /ollaya\.(cobanov\.)?dev/i
for (const dir of ['src', 'docs', 'content', 'public', 'bench-public']) {
  for (const file of await walk(join(root, dir))) {
    // src/generated/registry.ts mirrors the registry's manifests, whose blob URLs build.mjs checks against SITE_ORIGIN.
    if (/\.(png|ico|woff2?)$/.test(file) || file.includes(`${sep}generated${sep}`)) continue
    const m = hostPattern.exec(await readFile(file, 'utf8'))
    if (m) errors.push(`${relative(root, file)}: hard-coded host "${m[0]}" — use SITE_ORIGIN / {{SITE_ORIGIN}}`)
  }
}

// 4. Internal links ---------------------------------------------------------------------------------
const redirects = (await readFile(join(dist, '_redirects'), 'utf8').catch(() => ''))
  .split('\n')
  .filter((l) => l.trim() && !l.trim().startsWith('#'))
  .map((l) => l.trim().split(/\s+/)[0])

function resolves(urlPath) {
  const p = decodeURIComponent(urlPath)
  if (p === '/') return fileSet.has('index.html')
  const rel = p.slice(1).replace(/\/$/, '')
  return fileSet.has(rel) || fileSet.has(`${rel}.html`) || fileSet.has(`${rel}/index.html`) || redirects.includes(p)
}

let pages = 0
for (const file of files.filter((f) => f.endsWith('.html'))) {
  pages++
  const html = await readFile(join(dist, file), 'utf8')
  if (!html.startsWith('<!DOCTYPE html>')) errors.push(`dist/${file}: missing doctype`)
  if (/\{\{SITE_(ORIGIN|HOST)\}\}/.test(html)) errors.push(`dist/${file}: unreplaced {{SITE_ORIGIN}}/{{SITE_HOST}}`)
  if (html.includes('{{result ')) errors.push(`dist/${file}: unreplaced {{result …}} reference`)
  if (/coming soon|pre-release|illustrative output|being built/i.test(html.replace(/<[^>]+>/g, ' '))) {
    errors.push(`dist/${file}: still says "coming soon", "pre-release", "illustrative output" or "being built"`)
  }
  if (/htmx/i.test(html)) errors.push(`dist/${file}: still references htmx`)
  for (const [, url] of html.matchAll(/(?:href|src)="(\/(?!\/)[^"#?]*)[^"]*"/g)) {
    if (!resolves(url)) errors.push(`dist/${file}: broken internal link ${url}`)
  }
}

// 5. Installer -------------------------------------------------------------------------------------
const installer = await readFile(join(root, '..', 'scripts', 'install.sh')).catch(() => null)
const served = await readFile(join(dist, 'install.sh')).catch(() => null)
if (!installer || !served || !installer.equals(served)) errors.push('dist/install.sh is not a copy of ../scripts/install.sh')

// 6. Manifests by digest ------------------------------------------------------------------------------
// The digests are what `ollaya pull <name>@sha256:<hex>` asks for; docs/decisions/0006-manifest-retention.md.
const DIGEST_FILE = /^sha256-[0-9a-f]{64}$/
let tagManifests = 0
let digestManifests = 0
for (const file of files.filter((f) => /^v2\/[^/]+\/[^/]+\/manifests\/[^/]+$/.test(f))) {
  const name = file.split('/').pop()
  const bytes = await readFile(join(dist, file))
  const hex = createHash('sha256').update(bytes).digest('hex')
  if (DIGEST_FILE.test(name)) {
    digestManifests++
    if (name !== `sha256-${hex}`) errors.push(`dist/${file} is sha256:${hex}, not the manifest its name says`)
    continue
  }
  tagManifests++
  const kept = file.replace(/[^/]+$/, `sha256-${hex}`)
  const twin = await readFile(join(dist, kept)).catch(() => null)
  if (!twin?.equals(bytes)) {
    errors.push(`dist/${file} has no copy by digest (${kept}); run node scripts/archive-manifests.mjs and commit it`)
  }
}
if (
  tagManifests &&
  !redirects.includes('/v2/:namespace/:model/manifests/sha256:*')
) {
  errors.push('dist/_redirects does not serve /v2/<namespace>/<model>/manifests/sha256:<hex>')
}

if (errors.length) {
  console.error(`check: ${errors.length} problem(s)\n  - ${errors.join('\n  - ')}`)
  process.exit(1)
}
console.log(
  `check: ok (${files.length} files, ${pages} pages, all internal links resolve, no hard-coded hosts; ` +
    `${tagManifests} tag manifests, ${digestManifests} manifests by digest)`,
)
