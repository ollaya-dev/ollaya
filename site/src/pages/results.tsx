/**
 * /results: everything we measure, drawn from ../results (src/data/results.ts). Accuracy and
 * calibration on Bespoke Labs' public benchmark (with Ollama on the same GPU), speed of every
 * model on every machine, and parity with the authors' own code on each device.
 *
 * Colour carries data only (charts.tsx): Ollaya blue against the other server's orange, GPUs in
 * blues and CPUs in oranges, Vulkan purple, a parity check green or red. Everything else is ink.
 */
import type { Child } from 'hono/jsx'
import { Bars, DotPlot, Figure, Heatmap, Responsive, Scatter, toneVar, type DotSeries, type SeriesKey, type Tone } from '../components/charts'
import { textLink } from '../components/ui'
import { benchUrl } from '../lib/bench'
import {
  latencyRuns,
  latencySeries,
  machines,
  parityRuns,
  publicBench,
  publicRuns,
  runs,
  shortName,
  type CrossDevice,
  type DeviceSeries,
  type Machine,
  type ParityResult,
  type PublicResult,
} from '../data/results'
import { benchEdition, measured } from '../data/numbers'
import { published } from '../data/catalog'

const SUBSET_LABELS: Record<string, string> = {
  'vitaminc-dev': 'VitaminC',
  'massive-en-US': 'MASSIVE en',
  'massive-de-DE': 'MASSIVE de',
  boolq: 'BoolQ',
  squad2: 'SQuAD 2',
  paws: 'PAWS',
  multinli: 'MultiNLI',
  civil_comments: 'Civil Comments',
  aegis2: 'Aegis 2',
  helpsteer2: 'HelpSteer 2',
  'summeval-relevance': 'SummEval rel.',
  'summeval-consistency': 'SummEval cons.',
  pubmedqa: 'PubMedQA',
}

const nf = new Intl.NumberFormat('en-US')

const sections = [
  { id: 'bench', label: 'Test bench' },
  { id: 'accuracy', label: 'Accuracy and speed' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'datasets', label: 'By dataset' },
  { id: 'speed', label: 'Speed' },
  { id: 'windows', label: 'CUDA and Vulkan' },
  { id: 'parity', label: 'Parity' },
  { id: 'data', label: 'Raw data' },
]

const label = (r: PublicResult) => (r.server === 'ollama' ? `${r.model} (Ollama)` : r.model)
const tone = (r: PublicResult): Tone => (r.server === 'ollama' ? 'them' : 'us')
const rejectedNote = (r: PublicResult) => (r.errors > r.records * 0.05 ? `${Math.round((r.errors / r.records) * 100)} % rejected` : undefined)

/** The two servers of the public benchmark, as the legends name them. */
function serverKeys(): SeriesKey[] {
  const v = publicBench?.servers ?? {}
  return [
    { label: `Ollaya ${v.ollaya?.split(' ')[0] ?? ''}`.trim(), tone: 'us' },
    { label: `Ollama ${v.ollama ?? ''}`.trim(), tone: 'them', mark: 'ring' },
  ]
}

/** A GPU in blue, a CPU in orange, Vulkan in purple. */
function deviceTone(s: DeviceSeries): Tone {
  return s.device === 'vulkan' ? 'vulkan' : s.device === 'cpu' ? 'cpu' : 'gpu'
}

function Section({ id, title, lead, body, children }: { id: string; title: string; lead: string; body?: Child; children: Child }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} class="scroll-mt-24" data-section>
      <h2 id={`${id}-title`} class="text-base font-semibold text-fg">
        {title}
      </h2>
      <p class="mt-2 max-w-3xl text-2xl leading-tight font-medium tracking-tight text-fg md:text-3xl">{lead}</p>
      {body ? <p class="mt-4 max-w-3xl text-base text-body md:text-lg">{body}</p> : null}
      <div class="mt-10">{children}</div>
    </section>
  )
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div class="px-6 py-6 sm:px-7">
      <p class="text-4xl font-semibold tracking-tight whitespace-nowrap text-fg tabular-nums lg:text-5xl">{value}</p>
      <p class="mt-2 text-sm text-muted">{label}</p>
    </div>
  )
}

/** The headline numbers, counted from the runs. */
function totals() {
  const answers = publicRuns.reduce((n, r) => n + r.results.reduce((m, x) => m + x.records, 0), 0)
  let requests = 0
  for (const r of latencyRuns) {
    const p = (r as unknown as { protocol?: { warm?: number; n?: number } }).protocol ?? {}
    requests += r.results.filter((x) => !x.error).length * ((p.warm ?? 0) + (p.n ?? 0) + 1)
  }
  const parityQuestions = parityRuns.reduce((n, r) => n + r.results.reduce((m, x) => m + (x.questions ?? 0), 0), 0)
  const models = new Set<string>()
  for (const r of publicRuns) for (const x of r.results) if (x.server === 'ollaya') models.add(x.model)
  for (const r of latencyRuns) for (const x of r.results) if (!x.error) models.add(shortName(x.model))
  return { answers, requests, parityQuestions, models: models.size }
}

function Overview() {
  const t = totals()
  return (
    <div class="grid overflow-hidden rounded-2xl border border-line sm:grid-cols-2 lg:grid-cols-4">
      <Stat value={nf.format(t.answers)} label="benchmark answers scored against human labels" />
      <div class="border-t border-line sm:border-t-0 sm:border-l">
        <Stat value={nf.format(t.parityQuestions)} label="questions checked against the authors' own code" />
      </div>
      <div class="border-t border-line lg:border-t-0 lg:border-l">
        <Stat value={`${t.models}`} label="model tags measured" />
      </div>
      <div class="border-t border-line sm:border-l lg:border-t-0">
        <Stat value={nf.format(t.requests)} label="requests timed on our GPUs and CPUs" />
      </div>
    </div>
  )
}

/** The GPU the public benchmark ran on, as pages name it ("RTX 5090"). */
const publicGpu = machines[publicBench?.machine ?? '']?.headline ?? 'GPU'
const LOG_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

/**
 * The accuracy-against-latency chart's data (the page and the README's .svg share it): one point per model and
 * server, its most accurate tag, so the chart stays readable; the table under it lists every tag. The axes
 * cover the points, so none sits outside them.
 */
export function accuracyScatter() {
  const best = new Map<string, PublicResult>()
  for (const r of publicBench?.results ?? []) {
    if (r.errors >= r.records * 0.05) continue
    const key = `${r.server}/${shortName(r.model).split(':')[0]}`
    const b = best.get(key)
    if (!b || r.macro_accuracy > b.macro_accuracy || (r.macro_accuracy === b.macro_accuracy && r.median_ms < b.median_ms)) best.set(key, r)
  }
  const results = [...best.values()]
  const xs = results.map((r) => r.median_ms)
  const ys = results.map((r) => r.macro_accuracy)
  const lo = Math.floor(Math.min(...ys) * 10)
  const hi = Math.ceil(Math.max(...ys) * 10)
  return {
    points: results.map((r) => ({ label: r.model, x: r.median_ms, y: r.macro_accuracy, tone: tone(r), mark: r.server === 'ollama' ? ('ring' as const) : undefined })),
    // The 1-2-5 ticks from the one at or under the fastest point to the one at or over the slowest.
    xTicks: LOG_TICKS.filter((t, i) => (LOG_TICKS[i + 1] ?? Infinity) > Math.min(...xs) && (LOG_TICKS[i - 1] ?? 0) < Math.max(...xs)),
    yTicks: Array.from({ length: hi - lo + 1 }, (_, i) => (lo + i) / 10),
    xTitle: 'median latency per question (ms), log scale',
    yTitle: 'accuracy',
    label: `Accuracy against median latency on Bespoke Labs' public benchmark, ${publicGpu}, Ollaya and Ollama: the most accurate tag of each model`,
    legend: serverKeys(),
  }
}

/** The speed chart's data: one row per model, one series per machine and device. */
export function speedDots() {
  const series = latencySeries()
  // Colour says GPU or CPU; the mark says which machine: the first machine's dots are filled,
  // the others' are rings, so two machines on the same device type stay apart where they overlap.
  const machineOrder = Object.keys(machines)
  const dots: DotSeries[] = series.map((s) => ({
    key: s.key,
    label: s.device === 'cpu' ? `${s.label} CPU` : s.label,
    tone: deviceTone(s),
    mark: machineOrder.indexOf(s.machine) === 0 ? 'dot' : 'ring',
  }))
  const models = new Set<string>()
  // An -fp16 tag that was measured but not shipped (kev:0.8b-fp16, ADR 0008) stays in the raw data only.
  for (const s of series) for (const m of s.p50.keys()) if (!m.endsWith('-fp16') || published(m)) models.add(m)
  const fastest = (m: string) => Math.min(...series.map((s) => s.p50.get(m) ?? Infinity))
  const rows = [...models]
    .sort((a, b) => fastest(a) - fastest(b))
    .map((m) => ({ label: m, values: Object.fromEntries(series.map((s) => [s.key, s.p50.get(m)])) }))
  return {
    rows,
    series: dots,
    // Up to 100 s: nimble:9b on a CPU takes about a minute; the scale clamps anything past its end.
    ticks: [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
    label: 'Median latency of five-question requests per model, on each GPU and CPU we measured',
  }
}

function AccuracySpeed() {
  if (!publicBench) return null
  const chart = accuracyScatter()
  return (
    <Section
      id="accuracy"
      title="Accuracy and speed"
      lead="Side by side with Ollama, on the same GPU."
      body={`Every tag of Ollama ${publicBench.servers.ollama ?? ''}'s nine decision models and Ollaya's, on Bespoke Labs' public benchmark: ${nf.format(3880)} human-labeled questions from 13 datasets, one ${publicGpu}.`}
    >
      <Figure
        title="Accuracy against latency"
        sub="up and to the left is better"
        legend={chart.legend}
        notes={
          <>
            The most accurate tag of each model; every tag is in the table below, as are models that reject more than 5 % of the
            questions. Mean accuracy over the 13 datasets; median request (one question) over HTTP.
          </>
        }
      >
        <Responsive wide={<Scatter {...chart} />} compact={<Scatter {...chart} compact />} />
      </Figure>
      <Versus rows={publicBench.results} />
    </Section>
  )
}

/** A long table folded under one line, so the charts above it stay the page's reading path. */
function Fold({ title, children }: { title: string; children: Child }) {
  return (
    <details class="group mt-10 rounded-2xl border border-line">
      <summary class="flex cursor-pointer list-none items-center justify-between gap-4 px-4 py-3 text-sm font-medium text-fg [&::-webkit-details-marker]:hidden">
        {title}
        <span class="text-muted transition-transform group-open:rotate-90" aria-hidden="true">›</span>
      </summary>
      <div class="overflow-x-auto border-t border-line">{children}</div>
    </details>
  )
}

/** Every model of both servers in one table: the public benchmark and typed-decisions, most accurate first. */
function Versus({ rows }: { rows: PublicResult[] }) {
  const typed = (r: PublicResult) => measured(r.server === 'ollama' ? `ollama/${r.model}` : shortName(r.model), 'typed.accuracy')
  const num = 'px-4 py-2 text-right whitespace-nowrap'
  return (
    <Fold title={`Every model in a table (${rows.length} rows, with typed-decisions)`}>
      <table class="w-full min-w-[40rem] text-left text-[13px] tabular-nums">
        <caption class="sr-only">Every model of Ollaya and Ollama on the public benchmark and typed-decisions, most accurate first</caption>
        <thead class="border-b border-line bg-subtle text-muted">
          <tr>
            <th scope="col" class="px-4 py-2.5 font-medium">Model</th>
            <th scope="col" class="px-4 py-2.5 font-medium">Server</th>
            <th scope="col" class="px-4 py-2.5 font-medium">Weights</th>
            <th scope="col" class="px-4 py-2.5 text-right font-medium">Accuracy</th>
            <th scope="col" class="px-4 py-2.5 text-right font-medium">ECE</th>
            <th scope="col" class="px-4 py-2.5 text-right font-medium">Median</th>
            <th scope="col" class="px-4 py-2.5 text-right font-medium">Typed-decisions</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-line">
          {[...rows]
            .sort((a, b) => b.macro_accuracy - a.macro_accuracy)
            .map((r) => {
              const t = typed(r)
              return (
                <tr>
                  <th scope="row" class="px-4 py-2 font-mono font-normal text-fg">
                    {r.model}
                  </th>
                  <td class="px-4 py-2" style={`color:${toneVar(tone(r))}`}>
                    {r.server === 'ollama' ? 'Ollama' : 'Ollaya'}
                  </td>
                  <td class="px-4 py-2 text-muted">{r.weights ?? "the authors'"}</td>
                  <td class={num}>
                    {r.macro_accuracy.toFixed(3)}
                    {rejectedNote(r) ? <span class="text-muted"> ({rejectedNote(r)})</span> : null}
                  </td>
                  <td class={num}>{r.pooled_ece.toFixed(3)}</td>
                  <td class={num}>{Math.round(r.median_ms)} ms</td>
                  <td class={num}>{t === undefined ? <span class="text-faint">·</span> : t.toFixed(3)}</td>
                </tr>
              )
            })}
        </tbody>
      </table>
      <p class="border-t border-line px-4 py-3 text-[13px] leading-relaxed text-muted">
        Typed-decisions: 2,000 questions, argmax against the majority label. Weights: Ollama's own GGUF files, or the authors'.
      </p>
    </Fold>
  )
}

function Calibration() {
  if (!publicBench) return null
  const rows = [...publicBench.results].sort((a, b) => a.pooled_ece - b.pooled_ece)
  const ollaya = publicBench.results.find((r) => r.server === 'ollaya' && r.model.startsWith('nimble'))
  const ollama = publicBench.results.find((r) => r.server === 'ollama' && r.model === 'nimble')
  return (
    <Section
      id="calibration"
      title="Calibration"
      lead="Confidence you can put a threshold on."
      body={
        ollaya && ollama
          ? `How far confidence is from how often the model is right. Same Nimble weights: ${ollaya.pooled_ece.toFixed(3)} on Ollaya (the author's temperature), ${ollama.pooled_ece.toFixed(3)} on Ollama (raw softmax).`
          : "Expected calibration error (ECE) measures how far a model's confidence is from how often it is right."
      }
    >
      <Figure title="Calibration error, all 3,880 questions" sub="ten-bin ECE of the top probability; lower is better" legend={serverKeys()}>
        <Bars
          label="Calibration error per model"
          max={Math.max(...rows.map((r) => r.pooled_ece))}
          rows={rows.map((r) => ({ label: label(r), value: r.pooled_ece, text: r.pooled_ece.toFixed(3), tone: tone(r), note: rejectedNote(r) }))}
        />
      </Figure>
    </Section>
  )
}

function ByDataset() {
  if (!publicBench) return null
  const columns = Object.keys(SUBSET_LABELS).map((k) => ({ key: k, label: SUBSET_LABELS[k]! }))
  const rows = [...publicBench.results].sort((a, b) => b.macro_accuracy - a.macro_accuracy)
  return (
    <Section
      id="datasets"
      title="Accuracy by dataset"
      lead="Where each model is strong, dataset by dataset."
      body="Each cell: the share of a dataset's questions answered like the human label."
    >
      <Figure title="Accuracy per dataset (%)" sub="darker is higher; rows sorted by the mean; Ollama's rows in orange">
        <Heatmap
          label="Accuracy per model and dataset"
          columns={columns}
          lo={0.3}
          hi={0.95}
          rows={rows.map((r) => ({ label: label(r), tone: tone(r), values: columns.map((c) => r.per_subset[c.key]?.accuracy ?? null) }))}
        />
      </Figure>
      <Fold title={`By question type, with rejected questions (${rows.length} rows)`}>
        <table class="w-full min-w-[40rem] text-left text-[13px] tabular-nums">
          <caption class="sr-only">Accuracy by question type, calibration error, latency and rejected questions per model</caption>
          <thead class="border-b border-line bg-subtle text-muted">
            <tr>
              <th scope="col" class="px-4 py-2.5 font-medium">Model</th>
              <th scope="col" class="px-4 py-2.5 text-right font-medium">Accuracy</th>
              <th scope="col" class="px-4 py-2.5 text-right font-medium">Choice</th>
              <th scope="col" class="px-4 py-2.5 text-right font-medium">Yes/no</th>
              <th scope="col" class="px-4 py-2.5 text-right font-medium">Score</th>
              <th scope="col" class="px-4 py-2.5 text-right font-medium">ECE</th>
              <th scope="col" class="px-4 py-2.5 text-right font-medium">Median</th>
              <th scope="col" class="px-4 py-2.5 text-right font-medium">Rejected</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-line">
            {rows.map((r) => (
              <tr class="text-body">
                <th scope="row" class="px-4 py-2 font-mono font-normal" style={r.server === 'ollama' ? `color:${toneVar('them')}` : undefined}>
                  {r.server === 'ollama' ? label(r) : <span class="text-fg">{label(r)}</span>}
                </th>
                <td class="px-4 py-2 text-right">{r.macro_accuracy.toFixed(3)}</td>
                <td class="px-4 py-2 text-right">{r.by_type.choice?.toFixed(3) ?? ''}</td>
                <td class="px-4 py-2 text-right">{r.by_type.noul?.toFixed(3) ?? ''}</td>
                <td class="px-4 py-2 text-right">{r.by_type.score?.toFixed(3) ?? ''}</td>
                <td class="px-4 py-2 text-right">{r.pooled_ece.toFixed(3)}</td>
                <td class="px-4 py-2 text-right whitespace-nowrap">{Math.round(r.median_ms)} ms</td>
                <td class="px-4 py-2 text-right">{r.errors ? nf.format(r.errors) : '0'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Fold>
    </Section>
  )
}

function Speed() {
  if (!latencySeries().length) return null
  const chart = speedDots()
  const gpuRun = latencyRuns.find((r) => r.device !== 'cpu') as unknown as { protocol?: { warm?: number; n?: number } } | undefined
  const cpuRun = latencyRuns.find((r) => r.device === 'cpu') as unknown as { protocol?: { warm?: number; n?: number } } | undefined
  return (
    <Section
      id="speed"
      title="Speed on our machines"
      lead="Every model, on every machine we have."
      body="Five questions on a short message over HTTP, one request at a time, a new message each time."
    >
      <Figure
        title="Five questions, median request"
        sub="further left is faster; GPUs in blue, CPUs in orange, one mark per machine"
        legend={chart.series}
        notes={`Linux under WSL2. Each model: one load, ${gpuRun?.protocol?.warm ?? 5} untimed requests, then ${gpuRun?.protocol?.n ?? 20} timed ones${cpuRun ? ` (on the CPU ${cpuRun.protocol?.warm ?? 2} and ${cpuRun.protocol?.n ?? 10})` : ''}. A missing dot: the model did not run there or is still being measured.`}
      >
        <Responsive wide={<DotPlot {...chart} />} compact={<DotPlot {...chart} compact />} />
      </Figure>
    </Section>
  )
}

function WindowsGpus() {
  const run = parityRuns.find((r) => r.results.some((x) => x.device === 'vulkan'))
  if (!run) return null
  const models = [...new Set(run.results.filter((x) => x.device === 'vulkan').map((x) => x.model))]
  const rows = models.flatMap((m) =>
    (['cuda', 'vulkan'] as const).flatMap((d) => {
      const r = run.results.find((x) => x.model === m && x.device === d)
      return r?.p50_ms ? [{ label: m, note: d === 'cuda' ? 'CUDA' : 'Vulkan', value: r.p50_ms, text: `${Math.round(r.p50_ms)} ms`, tone: (d === 'cuda' ? 'gpu' : 'vulkan') as Tone }] : []
    }),
  )
  return (
    <Section
      id="windows"
      title="Windows: CUDA and Vulkan"
      lead="Vulkan brings GGUF models to any GPU."
      body="Vulkan runs on NVIDIA, AMD and Intel GPUs without the 1.4 GB CUDA libraries. Both backends match llama.cpp on every question; CUDA is faster."
    >
      <Figure
        title={`Five questions, median, ${machines[run.machine ?? '']?.headline ?? 'GPU'} under Windows`}
        sub="the runtime without HTTP; lower is better"
        legend={[
          { label: 'CUDA', tone: 'gpu' },
          { label: 'Vulkan', tone: 'vulkan' },
        ]}
      >
        <Bars label="Latency on CUDA and Vulkan" max={Math.max(...rows.map((r) => r.value))} rows={rows} />
      </Figure>
    </Section>
  )
}

const DEVICE_ORDER = ['cuda', 'vulkan', 'mlx', 'metal', 'cpu', 'coreml']
const DEVICE_LABEL: Record<string, string> = { cuda: 'CUDA', vulkan: 'Vulkan', mlx: 'Apple GPU (MLX)', metal: 'Metal', cpu: 'CPU', coreml: 'Core ML' }

function Parity() {
  // Runtime parity of library models (tags like `kev:4b`) on the devices Ollaya runs on; export
  // checks, research checkpoints and rejected providers stay in the raw data.
  const entries: ParityResult[] = parityRuns.flatMap((r) =>
    r.results.filter((x) => x.kind !== 'export' && DEVICE_ORDER.includes(x.device) && x.pass !== null && x.pass !== undefined && /^[a-z0-9-]+:[\w.-]+$/.test(x.model) && (!x.model.endsWith('-fp16') || published(x.model))),
  )
  if (!entries.length) return null
  const devices = DEVICE_ORDER.filter((d) => entries.some((e) => e.device === d))
  const models = [...new Set(entries.map((e) => e.model))].sort()
  const cell = (m: string, d: string) => {
    const es = entries.filter((e) => e.model === m && e.device === d)
    if (!es.length) return null
    const pass = es.some((e) => e.pass === true)
    const q = Math.max(...es.map((e) => e.questions ?? e.decisions_same ?? 0))
    return { pass, q }
  }
  const crossRun = parityRuns.find((r) => r.cross_device)
  const cross: CrossDevice[] = crossRun?.cross_device?.results ?? []
  return (
    <Section
      id="parity"
      title="Parity"
      lead="The same answers as the authors' own code."
      body="Before a model ships, every decision must match the authors' code (or llama.cpp for GGUF), each option score within 0.001. -fp16 tags: at least 99.5 % of decisions the same."
    >
      <div class="overflow-x-auto rounded-2xl border border-line">
        <table class="w-full min-w-[32rem] text-left text-[13px] tabular-nums">
          <caption class="sr-only">Runtime parity per model and device: passed or outside the gate, with the number of questions compared</caption>
          <thead class="border-b border-line bg-subtle text-muted">
            <tr>
              <th scope="col" class="px-4 py-2.5 font-medium">Model</th>
              {devices.map((d) => (
                <th scope="col" class="px-4 py-2.5 font-medium">
                  {DEVICE_LABEL[d] ?? d}
                </th>
              ))}
            </tr>
          </thead>
          <tbody class="divide-y divide-line">
            {models.map((m) => (
              <tr>
                <th scope="row" class="px-4 py-2 font-mono font-normal text-fg">
                  {m}
                </th>
                {devices.map((d) => {
                  const c = cell(m, d)
                  if (!c) return <td class="px-4 py-2 text-faint">·</td>
                  return (
                    <td class="px-4 py-2 whitespace-nowrap">
                      <span class="font-medium" style={`color:${toneVar(c.pass ? 'pass' : 'fail')}`}>
                        {c.pass ? '✓' : '✗'}
                        <span class="hidden sm:inline">{c.pass ? ' passed' : ' outside'}</span>
                      </span>
                      {c.q ? <span class="text-muted"> {nf.format(c.q)} q</span> : null}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p class="mt-4 max-w-3xl text-[13px] leading-relaxed text-muted">
        Questions compared per device. "Outside": over the gate, so that device does not run the model. Per family:{' '}
        <a href="https://github.com/ollaya-dev/ollaya/tree/main/docs/families" class={textLink}>
          docs/families
        </a>
        .
      </p>
      {cross.length ? (
        <div class="mt-14">
          <Figure
            title="Same prompt, different backends"
            sub={`largest difference in an option's log-probability against the ${machines[crossRun?.machine ?? '']?.headline ?? 'GPU'}'s CUDA numbers`}
            legend={[
              { label: 'CPU', tone: 'cpu' },
              { label: 'Vulkan', tone: 'vulkan' },
            ]}
            notes={crossRun?.cross_device?.note}
          >
            <Bars
              label="Difference between devices"
              max={Math.max(...cross.map((c) => c.max_logprob_diff))}
              rows={cross.map((c) => ({
                label: c.model,
                note: `${c.device}, ${c.decisions_same} of ${c.questions} decisions the same`,
                value: c.max_logprob_diff,
                text: c.max_logprob_diff.toFixed(2),
                tone: (c.device.startsWith('cpu') ? 'cpu' : 'vulkan') as Tone,
              }))}
            />
          </Figure>
        </div>
      ) : null}
    </Section>
  )
}

/** One of our machines as a spec column: what it is, then what we measure on it. */
function BenchColumn({ m }: { m: Machine }) {
  const gpu = m.gpus[0] ?? ''
  const rows: [string, string][] = [
    ['GPU', gpu.replace(/^NVIDIA GeForce /, '')],
    ['CPU', m.cpu],
    ['Memory', m.ram_gb ? `${m.ram_gb} GB` : ''],
    ['System', m.os],
    ['Driver', m.driver ?? ''],
  ]
  return (
    <div class="border-t-2 border-fg pt-5">
      <h3 class="text-3xl font-semibold tracking-tight text-fg">{m.headline ?? m.label}</h3>
      <p class="mt-1 text-sm text-muted">{m.label}</p>
      <dl class="mt-5 space-y-2.5 text-[13px] leading-snug">
        {rows
          .filter(([, v]) => v)
          .map(([k, v]) => (
            <div class="grid grid-cols-[4.5rem_minmax(0,1fr)] gap-x-3">
              <dt class="text-muted">{k}</dt>
              <dd class="text-body">{v}</dd>
            </div>
          ))}
      </dl>
      {m.uses?.length ? (
        <>
          <p class="mt-6 text-[13px] font-medium text-fg">Measured here</p>
          <ul class="mt-2 space-y-1.5 text-[13px] leading-snug text-body" role="list">
            {m.uses.map((u) => (
              <li class="flex gap-2">
                <span class="mt-[0.45rem] size-1 shrink-0 rounded-full bg-muted" aria-hidden="true"></span>
                {u}
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  )
}

function TestBench() {
  const ours = Object.values(machines).filter((m) => m.kind !== 'community')
  const community = Object.values(machines).filter((m) => m.kind === 'community')
  return (
    <Section
      id="bench"
      title="Test bench"
      lead="The machines behind these numbers."
      body="Every number here comes from these machines, with the files ollaya pull fetches."
    >
      <div class="grid gap-x-10 gap-y-12 md:grid-cols-3">
        {ours.map((m) => (
          <BenchColumn m={m} />
        ))}
      </div>
      <p class="mt-10 max-w-3xl text-[13px] leading-relaxed text-muted">
        Software: Ollaya {benchEdition.updates.at(-1) ?? benchEdition.ollaya}: ONNX Runtime with CUDA 13, llama.cpp for GGUF, MLX on the Apple GPU.
        {community.length ? (
          <>
            {' '}
            Community benches, measured by contributors with the same tools:{' '}
            {community.map((m, i) => (
              <>
                {i ? '; ' : ''}
                {m.label.replace(/ \(community\)$/, '')} ({m.cpu.replace(/ \(.*$/, '')}), by{' '}
                <a href={m.source} class={textLink}>
                  {m.by}
                </a>
              </>
            ))}
            .
          </>
        ) : null}
      </p>
    </Section>
  )
}

function RawData() {
  return (
    <Section id="data" title="Raw data" lead="Every number, as the tools wrote it.">
      <p class="max-w-3xl text-[15px] leading-relaxed text-body">
        Each chart on this page is drawn from these files. The tools that wrote them are in the repository (
        <span class="font-mono text-[13px]">convert/ollaya_convert/bench_public.py</span>, <span class="font-mono text-[13px]">bench_latency.py</span> and
        the parity examples), so you can run the same measurements on your own hardware.
      </p>
      <ul class="mt-6 space-y-1.5 font-mono text-[12px] sm:text-[13px]" role="list">
        {runs.map((r) => (
          <li>
            <a href={`/data/results/${r.file}`} class={`${textLink} break-all`}>
              {r.file}
            </a>
          </li>
        ))}
      </ul>
    </Section>
  )
}

export function ResultsPage() {
  const newest = runs
    .map((r) => r.date ?? '')
    .filter(Boolean)
    .sort()
    .at(-1)
  return (
    <>
      <header class="mx-auto max-w-6xl px-4 pt-10 md:px-6 md:pt-16">
        <h1 class="text-4xl leading-[1.05] font-medium tracking-tight text-fg md:text-5xl">Results</h1>
        <p class="mt-5 max-w-2xl text-lg text-body">
          Accuracy, calibration, speed and parity, measured on our own GPUs and CPUs, with the raw data.
        </p>
        {newest ? <p class="mt-4 text-sm text-muted">Newest measurement: {newest.slice(0, 10)}</p> : null}
        <p class="mt-6 max-w-2xl rounded-xl border border-line bg-subtle px-4 py-3 text-sm text-body">
          Scores and a page per model:{' '}
          <a href={benchUrl()} class="font-medium text-fg underline underline-offset-4">
            Ollaya Bench
          </a>
          .
        </p>
        <div class="mt-10">
          <Overview />
        </div>
      </header>
      <div class="mx-auto mt-20 max-w-6xl px-4 pb-24 md:mt-28 md:px-6 lg:grid lg:grid-cols-[11rem_minmax(0,1fr)] lg:gap-16">
        <nav aria-label="Sections" class="hidden lg:block">
          <ul class="sticky top-28 -ml-3.5 space-y-2.5 text-sm" data-scrollspy>
            {sections.map((s) => (
              <li>
                <a
                  href={`#${s.id}`}
                  class="block border-l-2 border-transparent pl-3 text-muted hover:text-fg aria-[current=true]:border-fg aria-[current=true]:font-medium aria-[current=true]:text-fg"
                >
                  {s.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <div class="min-w-0 space-y-24 md:space-y-32">
          <TestBench />
          <AccuracySpeed />
          <Calibration />
          <ByDataset />
          <Speed />
          <WindowsGpus />
          <Parity />
          <RawData />
        </div>
      </div>
    </>
  )
}
