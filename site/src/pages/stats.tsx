import { textLink } from '../components/ui'
import { catalog } from '../data/catalog'
import stats from '../data/stats.json'
import { GITHUB_URL } from '../site'

interface Week {
  /** The day after the week's last day (UTC). */
  end: string
  pulls: number
  pulls_by_model: Record<string, number>
  install_sh: number
  install_ps1: number
  stars: number | null
  release_downloads: number | null
}

const weeks = stats.weeks as unknown as Week[]
if (!weeks.length) throw new Error('src/data/stats.json has no weeks: run scripts/stats-snapshot.mjs')
const latest = weeks.at(-1)!
const lastGithub = [...weeks].reverse().find((w) => w.stars !== null)
const listed = new Set(catalog.map((m) => m.name))

const fmt = (n: number | null) => (n === null ? '–' : n.toLocaleString('en-US'))
const day = (iso: string, offset: number) => {
  const d = new Date(Date.parse(`${iso}T00:00:00Z`) + offset * 86400e3)
  return `${d.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' })} ${d.getUTCDate()}`
}
/** "Oct 16 – Oct 22" for the week before `end`. */
const range = (end: string) => `${day(end, -7)} – ${day(end, -1)}`

const cell = 'px-4 py-3 text-right tabular-nums'
const head = 'px-4 py-2.5 text-right font-medium'

function Tile({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div class="rounded-lg border border-line p-4">
      <p class="text-[13px] text-muted">{label}</p>
      <p class="mt-1 text-2xl font-medium tracking-tight text-fg tabular-nums">{value}</p>
      <p class="mt-1 text-[13px] text-muted">{note}</p>
    </div>
  )
}

export function StatsPage() {
  const top = Object.entries(latest.pulls_by_model)
  const max = top[0]?.[1] ?? 1
  return (
    <div class="mx-auto max-w-[52rem] px-4 pt-8 md:px-6 md:pt-12">
      <h1 class="text-[30px] leading-tight font-medium tracking-tight text-fg">Stats</h1>
      <p class="mt-3 text-lg text-body">How much Ollaya is used, week by week. The counts are requests, not people.</p>

      <div class="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Tile label="Model pulls" value={fmt(latest.pulls)} note={range(latest.end)} />
        <Tile label="Install script runs" value={fmt(latest.install_sh + latest.install_ps1)} note={range(latest.end)} />
        <Tile label="GitHub stars" value={fmt(lastGithub?.stars ?? null)} note={lastGithub ? `on ${day(lastGithub.end, -1)}` : ''} />
        <Tile
          label="Release downloads"
          value={fmt(lastGithub?.release_downloads ?? null)}
          note={lastGithub ? `every release, on ${day(lastGithub.end, -1)}` : ''}
        />
      </div>

      <section class="mt-12" aria-labelledby="weeks-title">
        <h2 id="weeks-title" class="text-base font-semibold text-fg">
          Week by week
        </h2>
        <div class="mt-3 overflow-x-auto rounded-lg border border-line">
          <table class="w-full text-sm">
            <thead class="bg-subtle text-[13px] text-muted">
              <tr>
                <th scope="col" class="px-4 py-2.5 text-left font-medium">
                  Week (UTC)
                </th>
                <th scope="col" class={head}>
                  Model pulls
                </th>
                <th scope="col" class={head}>
                  install.sh
                </th>
                <th scope="col" class={head}>
                  install.ps1
                </th>
                <th scope="col" class={head}>
                  Stars
                </th>
                <th scope="col" class={head}>
                  Release downloads
                </th>
              </tr>
            </thead>
            <tbody class="divide-y divide-line">
              {[...weeks].reverse().map((w) => (
                <tr>
                  <td class="px-4 py-3 whitespace-nowrap text-body">{range(w.end)}</td>
                  <td class={`${cell} text-fg`}>{fmt(w.pulls)}</td>
                  <td class={`${cell} text-body`}>{fmt(w.install_sh)}</td>
                  <td class={`${cell} text-body`}>{fmt(w.install_ps1)}</td>
                  <td class={`${cell} text-body`}>{fmt(w.stars)}</td>
                  <td class={`${cell} text-body`}>{fmt(w.release_downloads)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section class="mt-12" aria-labelledby="models-title">
        <h2 id="models-title" class="text-base font-semibold text-fg">
          Most pulled models, {range(latest.end)}
        </h2>
        <ul class="mt-4 space-y-2 text-sm" role="list">
          {top.map(([model, n]) => (
            <li class="grid grid-cols-[9rem_1fr_4rem] items-center gap-3">
              {listed.has(model) ? (
                <a href={`/library/${model}`} class={`font-mono ${textLink}`}>
                  {model}
                </a>
              ) : (
                <span class="font-mono text-fg">{model}</span>
              )}
              <span class="h-2 rounded-full bg-fill">
                <span class="block h-2 rounded-full bg-fg" style={`width:${((100 * n) / max).toFixed(1)}%`}></span>
              </span>
              <span class="text-right text-body tabular-nums">{fmt(n)}</span>
            </li>
          ))}
        </ul>
      </section>

      <section class="mt-12" aria-labelledby="method-title">
        <h2 id="method-title" class="text-base font-semibold text-fg">
          How these are counted
        </h2>
        <ul class="mt-3 list-disc space-y-2 pl-5 text-body">
          <li>
            <strong class="font-medium text-fg">Model pulls</strong> are the registry's manifest requests from the
            Ollaya CLI's own user agent, so crawlers and monitors are left out. One <code class="font-mono">ollaya pull</code>{' '}
            can fetch several manifests (a router fetches its targets), and a single busy machine or CI job can move a
            week.
          </li>
          <li>
            <strong class="font-medium text-fg">Install script runs</strong> are fetches of{' '}
            <code class="font-mono">install.sh</code> by curl or Wget and of <code class="font-mono">install.ps1</code>{' '}
            by PowerShell. Updates with <code class="font-mono">ollaya update</code> run the script too.
          </li>
          <li>
            Both come from Cloudflare's edge analytics, which samples busy traffic, so they are estimates. Ollaya has
            no telemetry: the CLI, the server and the desktop app report nothing about how they are used.
          </li>
          <li>
            <strong class="font-medium text-fg">Stars and release downloads</strong> are GitHub's totals on the
            week's last day. Release downloads count the archives and the desktop installers of every release, not
            their checksum files.
          </li>
          <li>
            The numbers are added each week by <code class="font-mono">site/scripts/stats-snapshot.mjs</code>, in the{' '}
            <a href={`${GITHUB_URL}/blob/main/site/src/data/stats.json`} class={textLink}>
              repository
            </a>
            .
          </li>
        </ul>
      </section>
    </div>
  )
}
