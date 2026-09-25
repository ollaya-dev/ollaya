// Builds ui/ into dist/ for Tauri: app.js (esbuild), app.css (Tailwind) and index.html.
//   node scripts/build-ui.mjs          the app
//   node scripts/build-ui.mjs --mock   the same page with a fake backend, to preview in a browser
import { spawnSync } from 'node:child_process'
import { cp, mkdir, rm } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { build } from 'esbuild'

const root = fileURLToPath(new URL('..', import.meta.url))
const dist = `${root}dist`
const mock = process.argv.includes('--mock')

await rm(dist, { recursive: true, force: true })
await mkdir(dist, { recursive: true })
await build({
  entryPoints: [`${root}ui/src/main.ts`],
  bundle: true,
  format: 'esm',
  target: 'es2022',
  minify: !mock,
  outfile: `${dist}/app.js`,
  define: { __MOCK__: String(mock) },
  logLevel: 'warning',
})
const tw = spawnSync(
  process.execPath,
  [`${root}node_modules/@tailwindcss/cli/dist/index.mjs`, '-i', `${root}ui/src/styles.css`, '-o', `${dist}/app.css`, '--minify'],
  { stdio: ['ignore', 'ignore', 'inherit'] },
)
if (tw.status !== 0) process.exit(tw.status ?? 1)
await cp(`${root}ui/index.html`, `${dist}/index.html`)
console.log(`ui: built dist/${mock ? ' (mock backend)' : ''}`)
