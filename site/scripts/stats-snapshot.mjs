// Adds one week of adoption numbers to src/data/stats.json, which /stats renders, so no number on that
// page is typed in by hand. Run it once a week (it replaces the entry for the same week):
//
//   node scripts/stats-snapshot.mjs [--end YYYY-MM-DD]     # the 7 days before --end (UTC), default today
//
// Model pulls and install-script runs come from Cloudflare's edge analytics for the ollaya.dev zone, through
// the `cf` CLI (logged in, with analytics read access). They are sampled, so they are estimates, and they count
// requests, not people: only the CLI's own user agent (`ollaya/<version>`), curl or Wget for install.sh and
// PowerShell for install.ps1, which leaves out crawlers and monitors. Stars and release downloads are GitHub's
// totals on the day the script runs, so only a run on a week's last day records them.

import { spawnSync } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const ZONE = '336bf5195ebf9fab2f55d70c5d3b5d29' // ollaya.dev
const REPO = 'ollaya-dev/ollaya'
const file = fileURLToPath(new URL('../src/data/stats.json', import.meta.url))

const i = process.argv.indexOf('--end')
const end = new Date(`${i > 0 ? process.argv[i + 1] : new Date().toISOString().slice(0, 10)}T00:00:00Z`)
if (Number.isNaN(end.getTime())) throw new Error('--end takes a date, YYYY-MM-DD')
const start = new Date(end.getTime() - 7 * 86400e3)

function sql(query) {
  const r = spawnSync('cf', ['analytics', 'sql', 'post', '--scope-zone-tag', ZONE, '--time-range-start', start.toISOString(),
    '--time-range-end', end.toISOString(), '--query', `${query} FORMAT TSV`], { encoding: 'utf8' })
  if (r.status !== 0) throw new Error(`cf analytics sql failed:\n${r.stdout}${r.stderr}`)
  return JSON.parse(r.stdout).split('\n').filter(Boolean).map((l) => l.split('\t'))
}

async function github(path) {
  const r = await fetch(`https://api.github.com/${path}`, { headers: { accept: 'application/vnd.github+json' } })
  if (!r.ok) throw new Error(`GitHub ${path}: ${r.status}`)
  return r.json()
}

// The model is the path segment before /manifests/ (`/v2/<model>/manifests/<tag>`).
const pulls = {}
for (const [path, n] of sql(`SELECT clientRequestPath AS p, sum(sampleInterval) AS req FROM events.httpRequests
  WHERE clientRequestPath LIKE '/v2/%/manifests/%' AND userAgent LIKE 'ollaya/%' AND edgeResponseStatus = 200 GROUP BY p`)) {
  const model = path.split('/manifests/')[0].split('/').at(-1)
  pulls[model] = (pulls[model] ?? 0) + Number(n)
}

const installs = { sh: 0, ps1: 0 }
for (const [path, n] of sql(`SELECT clientRequestPath AS p, sum(sampleInterval) AS req FROM events.httpRequests
  WHERE edgeResponseStatus = 200 AND ((clientRequestPath = '/install.sh' AND (userAgent LIKE 'curl/%' OR userAgent LIKE 'Wget%'))
  OR (clientRequestPath = '/install.ps1' AND userAgent LIKE '%PowerShell/%')) GROUP BY p`))
  installs[path === '/install.sh' ? 'sh' : 'ps1'] += Number(n)

// GitHub has only today's totals, so a week that ended before today gets none.
const today = end.toISOString().slice(0, 10) >= new Date().toISOString().slice(0, 10)
const repo = today ? await github(`repos/${REPO}`) : null
let downloads = null
for (let page = 1; today; page++) {
  const releases = await github(`repos/${REPO}/releases?per_page=100&page=${page}`)
  // Archives and installers only: the install scripts fetch a checksum (and a key for GPU packs) with each one.
  for (const r of releases)
    for (const a of r.assets) if (!/\.(sha256|key|txt)$/.test(a.name)) downloads = (downloads ?? 0) + a.download_count
  if (releases.length < 100) break
}

const week = {
  end: end.toISOString().slice(0, 10),
  pulls: Object.values(pulls).reduce((a, b) => a + b, 0),
  pulls_by_model: Object.fromEntries(Object.entries(pulls).sort((a, b) => b[1] - a[1]).slice(0, 10)),
  install_sh: installs.sh,
  install_ps1: installs.ps1,
  stars: repo?.stargazers_count ?? null,
  release_downloads: downloads,
}

const stats = JSON.parse(await readFile(file, 'utf8'))
stats.weeks = [...stats.weeks.filter((w) => w.end !== week.end), week].sort((a, b) => a.end.localeCompare(b.end))
await writeFile(file, `${JSON.stringify(stats, null, 2)}\n`)
console.log(`stats: week ending ${week.end}: ${week.pulls} pulls, ${week.install_sh} install.sh, ${week.stars} stars`)
