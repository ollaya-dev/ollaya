/**
 * Ollaya Bench pages: the leaderboard (one per device), a page per model, the methodology and the
 * data downloads. Everything renders at build time; bench.js only adds sorting to the tables.
 */
import type { Child } from 'hono/jsx'
import { CodeBlock } from '../components/ui'
import { machines } from '../data/results'
import { GITHUB_URL } from '../site'
import { ParetoChart, TaskBars } from './charts'
import {
  badgeUrl,
  type Device,
  deviceUrl,
  devices,
  edition,
  fmtMb,
  fmtMs,
  frontier,
  memoryOn,
  modelUrl,
  rows,
  type Row,
  type SpeedResult,
  scoredRows,
} from './data'
import { Flag, ModelLink, Mono, num3, pct1, rowFlags, ScoreBar, Section, Stat } from './ui'

const nf = new Intl.NumberFormat('en-US')
/** "docs/families/winnow.md:133-137" → the lines on GitHub. */
const citeUrl = (cite: string) => {
  const [file, lines] = cite.split(':')
  const [a, b] = (lines ?? '').split('-')
  return `${GITHUB_URL}/blob/main/${file}${a ? `#L${a}${b ? `-L${b}` : ''}` : ''}`
}
const QUESTIONS = 3880 + 2000

function dateLabel(iso: string | undefined): string {
  if (!iso) return ''
  const d = new Date(iso.length === 10 ? `${iso}T12:00:00Z` : iso)
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' })
}

const shortLicense = (l: string) =>
  l.startsWith('Apache-2.0 (') ? 'Apache-2.0 + Gemma' : l.replace('LFM Open License v1.0', 'LFM Open 1.0').replace(/ \([^)]*\)/g, '').replace(' and ', ' + ')

// ---------------------------------------------------------------------------------------------
// Leaderboard

function DeviceSwitch({ current }: { current: Device }) {
  return (
    <nav aria-label="Hardware" class="flex flex-wrap gap-1 rounded-md border border-line p-1">
      {devices.map((d) => (
        <a
          href={deviceUrl(d)}
          aria-current={d === current ? 'page' : undefined}
          class={`rounded-md px-3.5 py-1.5 text-sm ${d === current ? 'bg-btn font-medium text-btn-fg' : 'text-body hover:bg-fill hover:text-fg'}`}
        >
          {d.name}
          <span class={`ml-1.5 text-xs ${d === current ? 'opacity-70' : 'text-muted'}`}>{d.device === 'cpu' ? 'CPU' : 'GPU'}</span>
        </a>
      ))}
    </nav>
  )
}

function answers(d: Device) {
  const on = scoredRows.filter((r) => r.speed.get(d.slug)?.p50_ms !== undefined)
  const p50 = (r: Row) => r.speed.get(d.slug)!.p50_ms!
  const mem = (r: Row) => memoryOn(d, r.speed.get(d.slug))
  const gpu = d.device !== 'cpu'
  const budget = gpu ? 100 : 1000
  const memBudget = gpu ? 8 * 1024 : 4 * 1024
  const best = (list: Row[]) => list.slice().sort((a, b) => b.score!.score - a.score!.score)[0]
  return [
    { q: 'Highest score', r: best(on) },
    { q: `Best under ${gpu ? '100 ms' : '1 s'}`, r: best(on.filter((r) => p50(r) <= budget)) },
    { q: `Best in ${gpu ? '8 GB of VRAM' : '4 GB of RAM'}`, r: best(on.filter((r) => (mem(r) ?? Infinity) <= memBudget)) },
    { q: 'Fastest', r: on.slice().sort((a, b) => p50(a) - p50(b))[0] },
  ].filter((a): a is { q: string; r: Row } => !!a.r)
}

function LeaderTable({ d }: { d: Device }) {
  const front = frontier(d)
  const max = Math.max(10, Math.ceil(Math.max(...scoredRows.map((r) => r.score!.score_ci[1])) / 10) * 10)
  const unscored = rows.filter((r) => !r.score).sort((a, b) => a.name.localeCompare(b.name))
  const gpu = d.device !== 'cpu'
  const th = (key: string, label: Child, cls = '', title?: string) => (
    <th scope="col" class={`px-3 py-2.5 font-medium whitespace-nowrap ${cls}`} title={title}>
      <button type="button" data-sort={key} class="cursor-pointer underline-offset-4 hover:text-fg hover:underline">
        {label}
      </button>
    </th>
  )
  const cell = (r: Row) => {
    const s = r.speed.get(d.slug)
    const mem = memoryOn(d, s)
    return (
      <tr class="border-t border-line hover:bg-subtle" data-row>
        <td class="px-3 py-2.5 text-muted tabular-nums" data-v={r.score?.rank ?? 999}>
          {r.score ? r.score.rank : '–'}
        </td>
        <td class="px-3 py-2.5" data-v={r.name}>
          <ModelLink name={r.name} class="text-[13.5px]" />
          {front.has(r.name) ? <Flag title={`Nothing on the ${d.name} is both faster and higher scoring`}>Pareto</Flag> : null}
          {rowFlags(r)}
          {r.publisher ? <div class="mt-0.5 text-xs text-muted">{r.publisher}</div> : null}
        </td>
        <td class="px-3 py-2.5" data-v={r.score?.score ?? -999}>
          {r.score ? <ScoreBar score={r.score.score} ci={r.score.score_ci} max={max} accent={front.has(r.name)} /> : <span class="text-muted">–</span>}
        </td>
        <td class="px-3 py-2.5 text-right tabular-nums" data-v={r.public?.macro_accuracy ?? -1}>
          {pct1(r.public?.macro_accuracy)}
        </td>
        <td class="px-3 py-2.5 text-right whitespace-nowrap tabular-nums" data-v={r.typed?.accuracy ?? -1}>
          {pct1(r.typed?.accuracy)}
          {r.typed?.trained_on_split ? <span title="Trained on this dataset's train split"> *</span> : null}
        </td>
        <td class="px-3 py-2.5 text-right tabular-nums" data-v={r.public?.pooled_ece ?? 9}>
          {num3(r.public?.pooled_ece)}
        </td>
        <td class="px-3 py-2.5 text-right font-medium whitespace-nowrap text-fg tabular-nums" data-v={s?.p50_ms ?? 1e9}>
          {s?.error ? <span class="font-normal text-muted" title={s.error}>failed</span> : fmtMs(s?.p50_ms)}
        </td>
        <td class="px-3 py-2.5 text-right whitespace-nowrap tabular-nums" data-v={mem ?? 1e9}>
          {fmtMb(mem)}
        </td>
        <td class="px-3 py-2.5 text-right tabular-nums" data-v={paramsValue(r.params)}>
          {r.params ?? '–'}
        </td>
        <td class="px-3 py-2.5 text-xs whitespace-nowrap text-muted" data-v={r.license} title={r.license}>
          {shortLicense(r.license)}
        </td>
      </tr>
    )
  }
  return (
    <div class="overflow-x-auto rounded-md border border-line">
      <table class="w-full min-w-[920px] text-left text-[13px]" data-sortable>
        <thead class="bg-subtle text-xs text-muted">
          <tr>
            {th('rank', '#', 'w-10', 'Rank by the score’s 95 % interval: models share a rank when their intervals overlap')}
            {th('model', 'Model')}
            {th('score', 'Score', '', 'Chance-corrected accuracy over 14 tasks, 0 to 100, with its 95 % interval')}
            {th('public', 'Public', 'text-right', 'Macro accuracy on Bespoke Labs’ public benchmark (13 subsets, 3,880 questions), %')}
            {th('typed', 'Typed', 'text-right', 'Accuracy on typed-decisions (2,000 questions), %')}
            {th('ece', 'ECE', 'text-right', 'Calibration error on the public benchmark, lower is better')}
            {th('latency', 'Latency', 'text-right', `Median five-question request on the ${d.name}`)}
            {th('memory', gpu ? 'VRAM' : 'RAM', 'text-right', gpu ? 'GPU memory the model takes' : 'Peak resident memory of the runner')}
            {th('params', 'Params', 'text-right')}
            {th('license', 'License')}
          </tr>
        </thead>
        <tbody>
          {scoredRows.map(cell)}
          {unscored.map(cell)}
        </tbody>
      </table>
    </div>
  )
}

function paramsValue(p: string | null): number {
  if (!p) return 0
  const m = /([\d.]+)\s*([MB])/i.exec(p)
  if (!m) return 0
  return parseFloat(m[1]!) * (m[2]!.toUpperCase() === 'B' ? 1000 : 1)
}

export function LeaderboardPage({ device }: { device: Device }) {
  if (!edition) return <p class="p-8">No edition yet.</p>
  const nModels = rows.length
  const front = frontier(device)
  return (
    <>
      <section class="mx-auto max-w-6xl px-6 pt-12 md:px-10 md:pt-16">
        <p class="font-mono text-xs tracking-wide text-muted uppercase">
          Edition {edition.id} · {dateLabel(edition.date)} · Ollaya {edition.ollaya}
        </p>
        <h1 class="mt-3 max-w-3xl text-4xl leading-[1.05] font-medium tracking-tight text-balance text-fg md:text-5xl">
          Every open decision model, measured on real hardware.
        </h1>
        <p class="mt-4 max-w-2xl text-[17px] leading-relaxed text-body">
          Accuracy, calibration, speed and memory of {nModels} models on the GPUs and CPUs people own. Every model runs through
          the same released runtime, checked against its authors’ code, and every number links to its raw data.
        </p>
        <dl class="mt-8 grid max-w-3xl grid-cols-2 gap-6 sm:grid-cols-4">
          <Stat label="Models" value={nModels} />
          <Stat label="Tasks" value="14" detail="13 public datasets and typed-decisions" />
          <Stat label="Questions per model" value={nf.format(QUESTIONS)} detail="human and gold labels" />
          <Stat label="Devices" value={devices.length} detail={devices.map((d) => d.name).join(', ')} />
        </dl>
      </section>

      <section class="mx-auto mt-12 max-w-6xl px-6 md:px-10" aria-labelledby="board-title">
        <div class="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 id="board-title" class="text-xl font-semibold tracking-tight text-fg">
              Leaderboard on the {device.name}
            </h2>
            <p class="mt-1 text-sm text-muted">{device.kind}. Pick the hardware closest to yours.</p>
          </div>
          <DeviceSwitch current={device} />
        </div>

        <dl class="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {answers(device).map(({ q, r }) => {
            const s = r.speed.get(device.slug)
            return (
              <div class="rounded-md border border-line p-4">
                <dt class="text-xs text-muted">{q}</dt>
                <dd class="mt-1">
                  <ModelLink name={r.name} class="text-[15px] font-semibold" />
                </dd>
                <dd class="mt-0.5 text-xs text-muted tabular-nums">
                  score {r.score!.score.toFixed(1)} · {fmtMs(s?.p50_ms)} · {fmtMb(memoryOn(device, s))}
                </dd>
              </div>
            )
          })}
        </dl>

        <figure class="mt-6 rounded-md border border-line p-4 md:p-6">
          <figcaption class="flex flex-wrap items-baseline justify-between gap-2">
            <span class="text-sm font-medium text-fg">Score against latency</span>
            <span class="text-xs text-muted">
              <span class="mr-1.5 inline-block size-2.5 rounded-md bg-us align-[-1px]"></span>Pareto frontier ({front.size}): nothing is
              both faster and higher scoring
              <span class="mr-1.5 ml-4 inline-block size-2.5 rounded-md bg-faint align-[-1px]"></span>other models
            </span>
          </figcaption>
          <div class="mt-4 hidden sm:block">
            <ParetoChart device={device} />
          </div>
          <div class="mt-4 sm:hidden">
            <ParetoChart device={device} compact />
          </div>
        </figure>

        <div class="mt-6">
          <LeaderTable d={device} />
        </div>
        <p class="mt-3 max-w-4xl text-xs leading-relaxed text-muted">
          Score: chance-corrected accuracy over 14 tasks (0 is guessing, 100 is every answer right), with its 95 % interval;
          models whose intervals overlap share a rank. Public: Bespoke Labs’ public benchmark, 3,880 human-labeled questions.
          Typed: typed-decisions, 2,000 questions. ECE: calibration error, lower is better. Latency: median of{' '}
          {String(device.run.protocol.n ?? '')} warm five-question requests over HTTP. * trained on that dataset’s train split, so the
          score leaves it out.{' '}
          <a href="/methodology" class="text-body underline underline-offset-4 hover:text-fg">
            How we measure
          </a>
          .
        </p>
      </section>
    </>
  )
}

// ---------------------------------------------------------------------------------------------
// Model page

function Kv({ k, v }: { k: string; v: Child }) {
  return (
    <div class="flex justify-between gap-4 border-t border-line py-2 text-sm first:border-t-0">
      <dt class="text-muted">{k}</dt>
      <dd class="text-right text-fg tabular-nums">{v}</dd>
    </div>
  )
}

function SpeedCells({ s, d }: { s: SpeedResult | undefined; d: Device }) {
  if (s?.error)
    return (
      <td colspan={6} class="px-3 py-2 text-muted" title={s.error}>
        failed: {s.error.slice(0, 90)}
      </td>
    )
  return (
    <>
      <td class="px-3 py-2 text-right font-medium text-fg" title={s?.ollaya ? `Ollaya ${s.ollaya}` : undefined}>
        {fmtMs(s?.p50_ms)}
      </td>
      <td class="px-3 py-2 text-right">{fmtMs(s?.p90_ms)}</td>
      <td class="px-3 py-2 text-right">{fmtMs(s?.load_ms)}</td>
      <td class="px-3 py-2 text-right">{d.device === 'cpu' ? '–' : fmtMb(s?.vram_mb)}</td>
      <td class="px-3 py-2 text-right">{fmtMb(s?.ram_peak_mb)}</td>
      <td class="px-3 py-2 font-mono text-xs">{s?.precision ?? '–'}</td>
    </>
  )
}

export function ModelPage({ r, origin, main }: { r: Row; origin: string; main: string }) {
  const total = scoredRows.length
  const badge = `[![Ollaya Bench](${origin}${badgeUrl(r.name)})](${origin}${modelUrl(r.name)})`
  const dir = r.name.replace(':', '-')
  const reproduce = [
    `ollaya pull ${r.name}`,
    ``,
    `# typed-decisions; export the test split once with`,
    `# uv run python -m ollaya_convert.bench_typed export --out typed.jsonl`,
    `python -m ollaya_convert.bench_typed run --model ${r.name} \\`,
    `  --url http://127.0.0.1:11435 --data typed.jsonl --out typed/${dir}`,
    ``,
    `# Bespoke Labs' public benchmark; build its 13 subsets first`,
    `NIMBLE_SRC=nimble python -m ollaya_convert.bench_public run --model ${r.name} \\`,
    `  --url http://127.0.0.1:11435 --data public --out public/${dir}`,
    ``,
    `# speed and memory`,
    `python -m ollaya_convert.bench_latency --models ${r.name} --memory \\`,
    `  --machine my-machine --device cuda --out latency.json`,
  ].join('\n')
  return (
    <div class="mx-auto max-w-6xl px-6 pt-10 md:px-10">
      <nav aria-label="Breadcrumb" class="text-sm text-muted">
        <a href="/" class="underline-offset-4 hover:text-fg hover:underline">
          Leaderboard
        </a>{' '}
        / <span class="text-body">{r.name}</span>
      </nav>
      <div class="mt-4 grid gap-10 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div class="min-w-0">
          <h1 class="font-mono text-3xl font-semibold tracking-tight text-fg md:text-4xl">
            {r.name}
            {rowFlags(r)}
          </h1>
          <p class="mt-2 text-sm text-muted">
            {r.title}
            {r.publisher ? ` · ${r.publisher}` : ''} · {r.params ?? '–'} · {r.runtime} · {r.license}
          </p>

          {r.score ? (
            <dl class="mt-8 grid grid-cols-2 gap-6 sm:grid-cols-4">
              <Stat label="Score" value={r.score.score.toFixed(1)} detail={`95 % interval ${r.score.score_ci[0].toFixed(1)}–${r.score.score_ci[1].toFixed(1)}`} />
              <Stat label="Rank" value={`${r.score.rank}`} detail={`of ${total} scored models`} />
              <Stat label="Public benchmark" value={pct1(r.public?.macro_accuracy)} detail="macro accuracy, %" />
              <Stat label="typed-decisions" value={pct1(r.typed?.accuracy)} detail={r.typed?.trained_on_split ? 'trained on its train split' : 'accuracy, %'} />
            </dl>
          ) : (
            <p class="mt-8 rounded-md border border-line p-4 text-sm text-muted">
              {r.builtin
                ? 'This model answers only its own built-in questions, so the accuracy tasks do not apply. Its speed and memory are below.'
                : 'No score in this edition.'}
            </p>
          )}

          {r.score ? (
            <section class="mt-12" aria-labelledby="tasks-title">
              <h2 id="tasks-title" class="text-lg font-semibold tracking-tight text-fg">
                By task
              </h2>
              <p class="mt-1 text-sm text-muted">
                Chance-corrected accuracy on each of the 14 tasks: 0 is guessing, 100 is every answer right.
                {r.score.trained_on.length ? ' * trained on its train split, so not in the score.' : ''}
              </p>
              <div class="mt-5 max-w-2xl">
                <TaskBars tasks={r.score.tasks} trainedOn={r.score.trained_on} />
              </div>
            </section>
          ) : null}

          {r.public || r.typed ? (
            <section class="mt-12" aria-labelledby="acc-title">
              <h2 id="acc-title" class="text-lg font-semibold tracking-tight text-fg">
                Accuracy and calibration
              </h2>
              <div class="mt-4 overflow-x-auto rounded-md border border-line">
                <table class="w-full min-w-[560px] text-left text-sm">
                  <thead class="bg-subtle text-xs text-muted">
                    <tr>
                      <th class="px-3 py-2 font-medium">Benchmark</th>
                      <th class="px-3 py-2 text-right font-medium">Accuracy</th>
                      <th class="px-3 py-2 text-right font-medium">Choice</th>
                      <th class="px-3 py-2 text-right font-medium">Noul</th>
                      <th class="px-3 py-2 text-right font-medium">Score</th>
                      <th class="px-3 py-2 text-right font-medium">ECE</th>
                      <th class="px-3 py-2 text-right font-medium">Brier</th>
                      <th class="px-3 py-2 text-right font-medium">Rejected</th>
                    </tr>
                  </thead>
                  <tbody class="tabular-nums">
                    {r.public ? (
                      <tr class="border-t border-line">
                        <td class="px-3 py-2">Public benchmark</td>
                        <td class="px-3 py-2 text-right font-medium text-fg">
                          {pct1(r.public.macro_accuracy)}
                          {r.public.macro_accuracy_ci ? (
                            <span class="ml-1 text-xs font-normal text-muted">
                              ({pct1(r.public.macro_accuracy_ci[0])}–{pct1(r.public.macro_accuracy_ci[1])})
                            </span>
                          ) : null}
                        </td>
                        <td class="px-3 py-2 text-right">{pct1(r.public.by_type.choice)}</td>
                        <td class="px-3 py-2 text-right">{pct1(r.public.by_type.noul)}</td>
                        <td class="px-3 py-2 text-right">{pct1(r.public.by_type.score)}</td>
                        <td class="px-3 py-2 text-right">{num3(r.public.pooled_ece)}</td>
                        <td class="px-3 py-2 text-right">{num3(r.public.brier)}</td>
                        <td class="px-3 py-2 text-right">
                          {nf.format(r.public.errors)} / {nf.format(r.public.records)}
                        </td>
                      </tr>
                    ) : null}
                    {r.typed ? (
                      <tr class="border-t border-line">
                        <td class="px-3 py-2">
                          typed-decisions{r.typed.trained_on_split ? <span title="Trained on this dataset's train split"> *</span> : null}
                        </td>
                        <td class="px-3 py-2 text-right font-medium text-fg">
                          {pct1(r.typed.accuracy)}
                          <span class="ml-1 text-xs font-normal text-muted">
                            ({pct1(r.typed.accuracy_ci[0])}–{pct1(r.typed.accuracy_ci[1])})
                          </span>
                        </td>
                        <td class="px-3 py-2 text-right">{pct1(r.typed.by_type.choice)}</td>
                        <td class="px-3 py-2 text-right">{pct1(r.typed.by_type.noul)}</td>
                        <td class="px-3 py-2 text-right">{pct1(r.typed.by_type.score)}</td>
                        <td class="px-3 py-2 text-right">{num3(r.typed.ece)}</td>
                        <td class="px-3 py-2 text-right">{num3(r.typed.brier)}</td>
                        <td class="px-3 py-2 text-right">
                          {r.typed.errors} / {r.typed.rows} requests
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
              <p class="mt-2 text-xs text-muted">
                Accuracy in %, with its 95 % bootstrap interval. A rejected request (the model can’t take the question, or the input
                is too long for it) counts as wrong.
              </p>
            </section>
          ) : null}

          <section class="mt-12" aria-labelledby="speed-title">
            <h2 id="speed-title" class="text-lg font-semibold tracking-tight text-fg">
              Speed and memory
            </h2>
            <div class="mt-4 overflow-x-auto rounded-md border border-line">
              <table class="w-full min-w-[640px] text-left text-sm">
                <thead class="bg-subtle text-xs text-muted">
                  <tr>
                    <th class="px-3 py-2 font-medium">Device</th>
                    <th class="px-3 py-2 text-right font-medium">p50</th>
                    <th class="px-3 py-2 text-right font-medium">p90</th>
                    <th class="px-3 py-2 text-right font-medium">Load</th>
                    <th class="px-3 py-2 text-right font-medium">VRAM</th>
                    <th class="px-3 py-2 text-right font-medium">RAM peak</th>
                    <th class="px-3 py-2 font-medium">Precision</th>
                  </tr>
                </thead>
                <tbody class="tabular-nums">
                  {devices.map((d) => {
                    const s = r.speed.get(d.slug)
                    return (
                      <tr class="border-t border-line">
                        <td class="px-3 py-2">
                          <a href={deviceUrl(d)} class="underline-offset-4 hover:underline">
                            {d.name}
                          </a>{' '}
                          <span class="text-xs text-muted">{d.device === 'cpu' ? 'CPU' : d.device.toUpperCase()}</span>
                        </td>
                        <SpeedCells s={s} d={d} />
                      </tr>
                    )
                  })}
                  {r.variants.flatMap((v) =>
                    devices
                      .filter((d) => v.speed.has(d.slug))
                      .map((d) => (
                        <tr class="border-t border-line">
                          <td class="px-3 py-2">
                            <Mono>{v.name}</Mono> <span class="text-xs text-muted">{d.name}</span>
                          </td>
                          <SpeedCells s={v.speed.get(d.slug)} d={d} />
                        </tr>
                      )),
                  )}
                </tbody>
              </table>
            </div>
            {r.variants.map((v) => (
              <p class="mt-3 max-w-2xl text-sm text-muted">
                <Mono>{v.name}</Mono> is the same model computing in fp16 on the GPU, not ranked separately. Its decisions agree with{' '}
                <Mono>{r.name}</Mono>'s on at least 99.5 % of the parity questions
                {v.typed && r.typed ? (
                  <>
                    ; typed-decisions {pct1(v.typed.accuracy)} against {pct1(r.typed.accuracy)}
                  </>
                ) : null}
                {v.typed?.ollaya ? <>, measured with Ollaya {v.typed.ollaya}</> : null}.
              </p>
            ))}
            <p class="mt-2 text-xs text-muted">
              Five-question requests over HTTP, one at a time, each a different short message. VRAM: the rise of the GPU’s used
              memory while the model is loaded. RAM peak: the runner process’s peak resident memory.
            </p>
          </section>

          <section class="mt-12" aria-labelledby="parity-title">
            <h2 id="parity-title" class="text-lg font-semibold tracking-tight text-fg">
              Parity with the authors’ code
            </h2>
            {r.parity.length ? (
              <div class="mt-4 overflow-x-auto rounded-md border border-line">
                <table class="w-full min-w-[560px] text-left text-sm">
                  <thead class="bg-subtle text-xs text-muted">
                    <tr>
                      <th class="px-3 py-2 font-medium">Device</th>
                      <th class="px-3 py-2 text-right font-medium">Same decision</th>
                      <th class="px-3 py-2 text-right font-medium">Max logit diff.</th>
                      <th class="px-3 py-2 font-medium">Result</th>
                      <th class="px-3 py-2 font-medium">Record</th>
                    </tr>
                  </thead>
                  <tbody class="tabular-nums">
                    {r.parity.map((p) => (
                      <tr class="border-t border-line">
                        <td class="px-3 py-2" title={(p as { reference?: string }).reference}>
                          {p.device_detail ?? p.device}
                        </td>
                        <td class="px-3 py-2 text-right">{p.decisions ?? (p.decisions_same !== undefined ? `${p.decisions_same}/${p.questions}` : '–')}</td>
                        <td class="px-3 py-2 text-right whitespace-nowrap">{p.max_logit_diff == null ? '–' : p.max_logit_diff.toExponential(1)}</td>
                        <td class="px-3 py-2">
                          {p.pass === true ? (
                            <span class="font-medium text-pass">pass{p.gate === 'fp16' ? ' (fp16 gate)' : ''}</span>
                          ) : p.pass === false ? (
                            <span class="font-medium text-fail">fail</span>
                          ) : (
                            <span class="text-muted">measured</span>
                          )}
                        </td>
                        <td class="px-3 py-2 font-mono text-xs">{p.cite ? <a href={citeUrl(p.cite)} class="underline-offset-4 hover:underline">{p.cite.replace(/^docs\//, '')}</a> : '–'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p class="mt-3 max-w-2xl text-sm text-muted">
                Its parity check is in the model’s family documentation in the repository (<Mono>docs/families/</Mono>); the
                compiled parity file does not list it yet.
              </p>
            )}
          </section>

          <section class="mt-12" aria-labelledby="repro-title">
            <h2 id="repro-title" class="text-lg font-semibold tracking-tight text-fg">
              Reproduce
            </h2>
            <p class="mt-1 max-w-2xl text-sm text-muted">
              The tools are in <Mono>convert/ollaya_convert</Mono> and need only Python’s standard library. Run them against a local
              Ollaya {edition?.ollaya} with the same model digest.
            </p>
            <div class="mt-4">
              <CodeBlock code={reproduce} lang="shell" />
            </div>
          </section>
        </div>

        <aside class="min-w-0 space-y-6 lg:pt-2">
          <div class="rounded-md border border-line p-4">
            <dl>
              <Kv k="Runtime" v={r.runtime} />
              <Kv k="Parameters" v={r.params ?? '–'} />
              <Kv k="License" v={<span class="text-xs">{r.license}</span>} />
              <Kv k="Digest" v={<span class="font-mono text-xs">{r.digest ? r.digest.slice(0, 12) : '–'}</span>} />
              <Kv k="Edition" v={edition ? edition.id : '–'} />
            </dl>
          </div>
          <div class="rounded-md border border-line p-4 text-sm">
            <p class="font-medium text-fg">Run it</p>
            <div class="mt-2">
              <CodeBlock code={`ollaya run ${r.name}`} lang="shell" />
            </div>
            <a href={`${main}/library/${r.name}`} class="mt-3 inline-block text-body underline underline-offset-4 hover:text-fg">
              Model page on {main.replace(/^https?:\/\//, '')}
            </a>
          </div>
          {r.score ? (
            <div class="rounded-md border border-line p-4 text-sm">
              <p class="font-medium text-fg">Badge</p>
              <img src={badgeUrl(r.name)} alt={`Ollaya Bench badge for ${r.name}`} class="mt-3" />
              <p class="mt-3 text-xs text-muted">For a model card or a README:</p>
              <div class="mt-2">
                <CodeBlock code={badge} lang="markdown" />
              </div>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------
// Methodology

const SUBSETS: [string, string, string, string, string][] = [
  ['vitaminc-dev', 'VitaminC dev', 'CC BY-SA 3.0', 'Choice, 3', '599'],
  ['massive-en-US', 'MASSIVE 1.1 test, en-US', 'CC BY 4.0', 'Choice, 18', '350'],
  ['massive-de-DE', 'MASSIVE 1.1 test, de-DE (same ids)', 'CC BY 4.0', 'Choice, 18', '350'],
  ['boolq', 'BoolQ validation', 'CC BY-SA 3.0', 'Noul', '300'],
  ['squad2', 'SQuAD 2.0 validation', 'CC BY-SA 4.0', 'Noul', '299'],
  ['paws', 'PAWS labeled_final test', 'Google PAWS, free for any purpose', 'Noul', '250'],
  ['multinli', 'MultiNLI dev matched', 'Mixed per genre, permissive', 'Choice, 3', '299'],
  ['civil_comments', 'Civil Comments test', 'CC0 1.0', 'Noul', '300'],
  ['aegis2', 'Nemotron Content Safety V2 test', 'CC BY 4.0', 'Noul', '250'],
  ['helpsteer2', 'HelpSteer2 validation', 'CC BY 4.0', 'Score, 5', '249'],
  ['summeval-relevance', 'SummEval, relevance', 'MIT', 'Score, 5', '240'],
  ['summeval-consistency', 'SummEval, consistency', 'MIT', 'Score, 5', '144'],
  ['pubmedqa', 'PubMedQA pqa_labeled', 'MIT', 'Choice, 3', '250'],
]

function Prose({ children }: { children: Child }) {
  return <div class="max-w-3xl space-y-4 text-[15px] leading-relaxed text-body">{children}</div>
}

export function MethodologyPage() {
  const flagged = Object.entries(edition?.trained_on ?? {})
  return (
    <div class="pb-8">
      <section class="mx-auto max-w-4xl px-6 pt-12 md:px-10">
        <p class="font-mono text-xs tracking-wide text-muted uppercase">Methodology · Edition {edition?.id}</p>
        <h1 class="mt-3 text-4xl font-medium tracking-tight text-fg">How we measure</h1>
        <p class="mt-4 max-w-2xl text-[17px] leading-relaxed text-body">
          One released runtime, every model in the configuration its authors ship, the same questions for each, and the same
          machine. This page is the whole protocol; the data page has every file it produced.
        </p>
      </section>

      <Section id="principles" title="Principles">
        <ul class="grid gap-5 sm:grid-cols-2">
          {[
            ['Measured, not modeled', 'Every latency and memory figure is measured on a named machine. No adjustment factors, no costs modeled from price lists.'],
            ['The authors’ model', 'Each model runs from its author’s unmodified weights and passes a parity check against the author’s own code before it is benchmarked. The number belongs to their model, not to a broken port.'],
            ['Reproducible', 'Every number comes with the runtime version, the model’s manifest digest, the machine and the command that produced it. The tools need only Python’s standard library.'],
            ['Frozen editions', 'An edition fixes the tasks, the protocol and the models. Scores compare within an edition; anything that changes goes in the changelog.'],
            ['Uncertainty shown', 'Scores carry 95 % bootstrap intervals, and models whose intervals overlap share a rank.'],
            ['No private pre-testing', 'We publish every version we measure. If Ollaya ever ships a model of its own, it runs under the same rules and is labeled as ours.'],
            ['Authors can respond', 'Each model runs with its author’s recommended temperature and precision. Corrections go through public GitHub issues.'],
            ['Contamination flagged', 'A model trained on a task’s train split carries a flag, and its score leaves that task out, so a result on seen data never counts as a held-out one.'],
          ].map(([t, d]) => (
            <li class="rounded-md border border-line p-4">
              <p class="font-medium text-fg">{t}</p>
              <p class="mt-1 text-sm leading-relaxed text-body">{d}</p>
            </li>
          ))}
        </ul>
      </Section>

      <Section id="score" title="The score">
        <Prose>
          <p>
            A model’s score is the mean, over 14 tasks with equal weight, of its <em>chance-corrected accuracy</em> on each task:
          </p>
          <p class="rounded-md bg-code px-4 py-3 font-mono text-sm text-fg">skill = 100 × (accuracy − chance) / (1 − chance)</p>
          <p>
            where <em>chance</em> is the mean of 1/K over the task’s questions (K options), so 0 is what guessing gets and 100 is
            every answer right. A rejected request (a question with more options than the model takes, an input longer than its
            context) is wrong. The 14 tasks are the 13 subsets of the public benchmark and typed-decisions. Equal weight means a
            450-question subset counts as much as typed-decisions’ 2,000 questions: the score rewards breadth.
          </p>
          <p>
            A model trained on a task’s train split is scored on the other 13 tasks: its result there is shown, flagged, but
            does not count.
          </p>
          <p>
            The 95 % interval is a bootstrap with 1,000 draws and a fixed seed: questions are resampled within each public subset,
            and typed-decisions states are resampled whole, their five questions together. The rank is one plus the number of
            models whose whole interval lies above this model’s.
          </p>
        </Prose>
      </Section>

      <Section id="tasks" title="The tasks" lead="5,880 questions per model: three answer types (choice, noul, score), human labels and gold distributions.">
        <div class="overflow-x-auto rounded-md border border-line">
          <table class="w-full min-w-[720px] text-left text-sm">
            <thead class="bg-subtle text-xs text-muted">
              <tr>
                <th class="px-3 py-2 font-medium">Task</th>
                <th class="px-3 py-2 font-medium">Source</th>
                <th class="px-3 py-2 font-medium">Type</th>
                <th class="px-3 py-2 text-right font-medium">Questions</th>
                <th class="px-3 py-2 font-medium">License</th>
              </tr>
            </thead>
            <tbody>
              <tr class="border-t border-line">
                <td class="px-3 py-2 font-mono text-xs">typed-decisions</td>
                <td class="px-3 py-2">
                  <a class="underline underline-offset-4" href="https://huggingface.co/datasets/LocalLLaMA/typed-decisions">
                    typed-decisions
                  </a>{' '}
                  test split: 400 JSON states from four workflows (agent traces, customer service, invoices, security incidents), five
                  questions each
                </td>
                <td class="px-3 py-2 whitespace-nowrap">Choice, noul, score</td>
                <td class="px-3 py-2 text-right tabular-nums">2,000</td>
                <td class="px-3 py-2">Apache-2.0</td>
              </tr>
              {SUBSETS.map(([id, src, lic, type, n]) => (
                <tr class="border-t border-line">
                  <td class="px-3 py-2 font-mono text-xs">{id}</td>
                  <td class="px-3 py-2">{src}</td>
                  <td class="px-3 py-2 whitespace-nowrap">{type}</td>
                  <td class="px-3 py-2 text-right tabular-nums">{n}</td>
                  <td class="px-3 py-2">{lic}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Prose>
          <p class="mt-4">
            The 13 public subsets are{' '}
            <a class="underline underline-offset-4" href="https://github.com/bespokelabsai/nimble/blob/main/docs/PUBLIC_BENCHMARKS.md">
              Bespoke Labs’ public decision benchmark
            </a>
            : seeded, label-blind draws from human-labeled datasets, rebuilt byte for byte from their committed ids and checksums,
            and scored with Bespoke’s own runner. Each request carries one question. typed-decisions requests carry a state’s five
            questions together, as an application would send them; the answer is the argmax of the returned probabilities against
            the gold label.
          </p>
        </Prose>
      </Section>

      <Section id="calibration" title="Calibration">
        <Prose>
          <p>
            A decision model returns probabilities, and an application acts on them, so calibration matters as much as accuracy.
            ECE is the ten-bin expected calibration error of the top probability, pooled over every answered question of the public
            benchmark (the leaderboard’s column) and of typed-decisions (on each model’s page). Brier is the multiclass Brier score.
            Lower is better for both. Each model runs with the temperatures its authors fitted, as Ollaya ships it.
          </p>
        </Prose>
      </Section>

      <Section id="speed" title="Speed and memory">
        <Prose>
          <p>
            One model at a time is loaded on a server started for the run. Each model gets one cold load (timed by the server), five
            untimed warm-up requests, then 20 timed requests on the GPU (10 on a CPU, after two warm-ups). Every request is the
            five-question triage preset on a different short customer message, sent over HTTP to <Mono>/api/decide</Mono> by a
            client on the same machine; the time is the client’s, end to end. The leaderboard shows the median.
          </p>
          <p>
            Memory: on a GPU, the rise of the GPU’s used memory (sampled every 100 ms) over its idle level just before the load, at
            its highest while the model is loaded. On every device, the runner process’s peak resident memory. GGUF models map their
            weights from disk, so part of their resident memory is file cache the system can reclaim.
          </p>
          <p>
            Accuracy is measured once, on the GPU. Parity makes it device-independent: a model gives the same decisions on every
            device within the tolerances of its parity check. ONNX models run in fp16 on a GPU where their authors’ weights allow
            it and in fp32 on a CPU; their decisions agree on 99.1–99.6 % of questions, the difference being near-ties.
          </p>
        </Prose>
      </Section>

      <Section id="hardware" title="Hardware">
        <div class="grid gap-4 sm:grid-cols-2">
          {[...new Set(devices.map((d) => d.machine))].map((id) => {
            const m = machines[id]!
            return (
              <div class="rounded-md border border-line p-4 text-sm">
                <p class="font-medium text-fg">{m.label}</p>
                <dl class="mt-2">
                  <Kv k="GPU" v={m.gpus.join(', ')} />
                  <Kv k="CPU" v={m.cpu} />
                  <Kv k="RAM" v={m.ram_gb ? `${m.ram_gb} GB` : '–'} />
                  <Kv k="OS" v={m.os} />
                  <Kv k="Driver" v={m.driver ?? '–'} />
                </dl>
              </div>
            )
          })}
        </div>
      </Section>

      <Section id="flags" title="Flags">
        <Prose>
          <ul class="list-disc space-y-2 pl-5">
            {flagged.map(([m, tasks]) => (
              <li>
                <Mono>{m}</Mono> was trained on the train split of {tasks.join(', ')}; its score leaves that task out.
              </li>
            ))}
            <li>
              <Mono>laya</Mono> is a router: each request goes to <Mono>laya:en</Mono> or <Mono>laya:multilingual</Mono> by its language.
            </li>
            <li>
              <Mono>qwen3guard:0.6b</Mono> answers only its own built-in safety questions, so it has speed and memory here but no
              accuracy.
            </li>
          </ul>
        </Prose>
      </Section>

      <Section id="releases" title="New models and releases">
        <Prose>
          <p>
            An edition is not frozen until the next one. Each Ollaya release adds what it changed, measured on the same
            machine with the same protocol, and the row says which release measured it:
          </p>
          <ul class="list-disc space-y-2 pl-5">
            <li>A new model, or new weights: every suite, after it passes parity on the GPU and the CPU.</li>
            <li>
              A new graph or engine for a model: parity again. When every decision is the same as before, its accuracy
              stays and only its speed is measured again.
            </li>
            <li>
              A faster, lower-precision tag such as <Mono>kev:4b-fp16</Mono>: typed-decisions and speed. It is shown on its
              base model's page, not ranked: its decisions agree with the base model's on at least 99.5 % of the parity
              questions.
            </li>
          </ul>
          <p>
            A new edition, with every suite for every model, follows a change that moves decisions across the library, or
            a quarter. The repository's <Mono>docs/model-lifecycle.md</Mono> is the full procedure, and{' '}
            <Mono>docs/decisions/0009-measurement-per-release.md</Mono> the rule for what each change measures.
          </p>
        </Prose>
      </Section>

      <Section id="run-it-yourself" title="Run it yourself">
        <Prose>
          <p>
            The tools are in the repository and need only Python's standard library and an Ollaya install. On your own
            machine:
          </p>
          <pre class="overflow-x-auto rounded-md border border-line bg-subtle p-4 font-mono text-[12px] leading-relaxed text-fg">
            {`python -m ollaya_convert.bench_run --machine community-<gpu> --out runs/ \\
    --models kev:4b d1:3b --suites latency-cuda typed \\
    --typed-data typed-decisions-test.jsonl`}
          </pre>
          <p>
            It starts a server, pulls the models and resumes where it stopped. To publish your numbers, open a pull request
            that adds your machine to <Mono>results/machines.json</Mono> and the run files: they appear on Ollaya's results page
            under your machine's name. An edition ranks only its own machine's rows.
          </p>
        </Prose>
      </Section>

      <Section id="limits" title="Limits">
        <Prose>
          <ul class="list-disc space-y-2 pl-5">
            <li>Labels measure agreement with human annotators, not truth. Annotators disagree, and some subsets are small.</li>
            <li>All tasks are public. A model trained on a test split would look better than it is; we flag the cases we know of.</li>
            <li>Speed is single-client latency. Ollaya serves one request per model at a time, so it is not a throughput benchmark.</li>
            <li>This edition measures one GPU and one CPU. Results on other hardware follow the same ranking only roughly.</li>
          </ul>
        </Prose>
      </Section>

      <Section id="changelog" title="Changelog">
        <Prose>
          <p>
            <span class="font-medium text-fg">2026.10</span> · {dateLabel(edition?.date)}. First edition: {rows.length} models,
            Ollaya {edition?.ollaya}, the RTX 5090 desktop (GPU and CPU).
          </p>
          {(edition?.updates ?? []).map((u) => (
            <p>
              <span class="font-medium text-fg">Ollaya {u.release}</span> · {dateLabel(u.date)}. Measured {u.models.join(', ')}.
            </p>
          ))}
        </Prose>
      </Section>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------
// Data

export function DataPage() {
  const files = edition?.runs ?? []
  return (
    <div class="pb-8">
      <section class="mx-auto max-w-4xl px-6 pt-12 md:px-10">
        <p class="font-mono text-xs tracking-wide text-muted uppercase">Data · Edition {edition?.id}</p>
        <h1 class="mt-3 text-4xl font-medium tracking-tight text-fg">Every number, as the tools wrote it</h1>
        <p class="mt-4 max-w-2xl text-[17px] leading-relaxed text-body">
          The files behind every page of this site, unchanged. They are also in the repository under <Mono>results/</Mono>, with
          the tools that wrote them.
        </p>
      </section>
      <Section id="files" title="Files">
        <ul class="divide-y divide-line rounded-md border border-line">
          <li class="flex flex-wrap items-baseline justify-between gap-2 px-4 py-3 text-sm">
            <a href="/api/v1/leaderboard.json" class="font-mono text-fg underline underline-offset-4">
              /api/v1/leaderboard.json
            </a>
            <span class="text-muted">One row per model: score, accuracy, calibration, speed and memory on each device</span>
          </li>
          <li class="flex flex-wrap items-baseline justify-between gap-2 px-4 py-3 text-sm">
            <a href={`/data/editions/${edition?.id}.json`} class="font-mono text-fg underline underline-offset-4">
              editions/{edition?.id}.json
            </a>
            <span class="text-muted">The edition: its runs, models, scores and flags</span>
          </li>
          {files.map((f) => (
            <li class="flex flex-wrap items-baseline justify-between gap-2 px-4 py-3 text-sm">
              <a href={`/data/runs/${f}`} class="font-mono text-fg underline underline-offset-4">
                runs/{f}
              </a>
              <span class="text-muted">
                {f.includes('public-benchmark')
                  ? 'Public benchmark: per model and per subset'
                  : f.includes('typed-decisions')
                    ? 'typed-decisions: per model, type and workflow'
                    : f.endsWith('-cpu.json')
                      ? 'Speed and memory on the CPU'
                      : 'Speed and memory on the GPU'}
              </span>
            </li>
          ))}
        </ul>
        <p class="mt-4 max-w-3xl text-sm text-muted">
          Per-question outputs (every request and every answer, about 6,000 per model) are kept with the edition’s raw files and
          will be published as a dataset. The JSON API is static: it changes only when an edition does.
        </p>
      </Section>
    </div>
  )
}

export function BenchNotFound() {
  return (
    <div class="mx-auto max-w-2xl px-4 py-24 text-center">
      <p class="font-mono text-sm text-muted">404</p>
      <h1 class="mt-2 text-3xl font-semibold tracking-tight text-fg">Page not found</h1>
      <p class="mt-3 text-body">
        <a href="/" class="underline underline-offset-4">
          Back to the leaderboard
        </a>
      </p>
    </div>
  )
}

