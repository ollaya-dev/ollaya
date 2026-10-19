/**
 * Ollaya Bench data: one edition (results/editions/<id>.json) and the runs it names, joined with the
 * catalog (publisher, licence, runtime) and the parity record. Everything on the bench site comes
 * from here; nothing is typed in by hand.
 */
import { catalog, published, type Model } from '../data/catalog'
import { editionsRaw } from '../generated/results'
import { machines, parityRuns, runs, shortName, type Machine, type ParityResult, type PublicResult } from '../data/results'

export interface ScoreResult {
  model: string
  score: number
  score_ci: [number, number]
  tasks: Record<string, number>
  trained_on: string[]
}

export interface Edition {
  schema: 1
  id: string
  title: string
  date: string
  ollaya: string
  machine: string
  runs: string[]
  models: string[]
  score: { definition: string; interval: string; results: ScoreResult[] }
  trained_on: Record<string, string[]>
  /** Releases that added their runs to this edition (ADR 0009). */
  updates?: { release: string; date: string; runs: string[]; models: string[] }[]
}

export interface TypedResult {
  model: string
  digest?: string | null
  answered_by: string[]
  rows: number
  questions: number
  errors: number
  accuracy: number
  accuracy_ci: [number, number]
  by_type: Record<string, number | null>
  by_workflow: Record<string, number>
  ece: number | null
  brier: number | null
  soft_ce: number | null
  median_ms: number | null
  p95_ms: number | null
  trained_on_split: boolean
  ollaya?: string
}

export interface SpeedResult {
  model: string
  questions?: string
  load_ms?: number
  first_request_ms?: number
  answered_by?: string
  p50_ms?: number
  p90_ms?: number
  min_ms?: number
  eval_p50_ms?: number
  input_tokens?: number
  device?: string
  precision?: string
  digest?: string
  size_mb?: number
  ram_peak_mb?: number
  ram_mb?: number
  vram_mb?: number
  error?: string
  /** The Ollaya release that measured this row (an edition's own, or a later release's update). */
  ollaya?: string
}

export interface BenchPublic extends PublicResult {
  digest?: string | null
  macro_accuracy_ci?: [number, number]
}

/** One way to run the models: a machine and the device on it. */
export interface Device {
  /** URL slug: the GPU's or the CPU's name, e.g. "rtx-5090", "threadripper-3960x". */
  slug: string
  machine: string
  device: 'cuda' | 'cpu' | string
  /** Short name: "RTX 5090", "Threadripper 3960X". */
  name: string
  /** What it is: "NVIDIA GPU, CUDA", "CPU, 24 cores". */
  kind: string
  info: Machine
  run: { file: string; date: string; ollaya: string; protocol: Record<string, unknown>; note?: string }
  results: Map<string, SpeedResult>
}

export interface Row {
  name: string
  family: string
  model?: Model
  title: string
  publisher: string | null
  description: string
  params: string | null
  runtime: 'ONNX Runtime' | 'llama.cpp' | 'router'
  license: string
  digest: string | null
  score?: ScoreResult & { rank: number }
  public?: BenchPublic
  typed?: TypedResult
  speed: Map<string, SpeedResult>
  parity: ParityResult[]
  /** Faster tags of the same model (`-fp16`): shown under it, not ranked (ADR 0009). */
  variants: Row[]
  /** Answers only its own built-in questions, so no accuracy. */
  builtin: boolean
  router: boolean
}

const editions = editionsRaw as Edition[]
/** The newest edition; the site is built around it. */
export const edition: Edition | undefined = editions.at(-1)
export const allEditions = editions

type AnyRun = { file: string; suite: string; date?: string; machine?: string; device?: string; ollaya?: string; protocol?: Record<string, unknown>; note?: string; results: unknown[] }
const editionRuns = (runs as unknown as AnyRun[]).filter((r) => edition?.runs.includes(r.file))

const slugify = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')

function deviceName(m: Machine, device: string): { name: string; kind: string } {
  if (device === 'cpu') {
    const cpu = m.cpu.replace(/ \(.*$/, '').replace(/^(AMD Ryzen|Intel Core) /, '').replace(/ \d+-Core Processor$/, '')
    const cores = /\((\d+) cores/.exec(m.cpu)?.[1]
    return { name: cpu, kind: cores ? `CPU, ${cores} cores` : 'CPU' }
  }
  const gpu = (m.gpus[0] ?? 'GPU').replace(/^NVIDIA GeForce /, '').replace(/, .*$/, '')
  const mem = /, (\d+ GB)/.exec(m.gpus[0] ?? '')?.[1]
  return { name: gpu, kind: `NVIDIA GPU, ${device.toUpperCase()}${mem ? `, ${mem}` : ''}` }
}

const versionOf = (r: AnyRun) => r.ollaya ?? (r as { servers?: { ollaya?: string } }).servers?.ollaya ?? ''

/**
 * The devices this edition timed the models on, GPUs first. Releases add runs to an edition (ADR 0009): a later
 * run's row replaces an earlier one of the same model, and keeps the release that measured it.
 */
const latencyGroups = new Map<string, AnyRun[]>()
for (const r of editionRuns.filter((r) => r.suite === 'latency-triage')) {
  const key = `${r.machine}/${r.device}`
  latencyGroups.set(key, [...(latencyGroups.get(key) ?? []), r])
}
export const devices: Device[] = [...latencyGroups.values()]
  .map((group) => {
    const r = group[0]!
    const info = machines[r.machine ?? '']!
    const { name, kind } = deviceName(info, r.device ?? '')
    const results = new Map<string, SpeedResult>()
    for (const g of group) for (const x of g.results as SpeedResult[]) results.set(shortName(x.model), { ...x, ollaya: versionOf(g) })
    return {
      slug: slugify(name),
      machine: r.machine ?? '',
      device: r.device ?? '',
      name,
      kind,
      info,
      run: { file: r.file, date: r.date ?? '', ollaya: r.ollaya ?? '', protocol: r.protocol ?? {}, note: r.note },
      results,
    }
  })
  .sort((a, b) => Number(a.device === 'cpu') - Number(b.device === 'cpu'))

type RunOf<T> = Omit<AnyRun, 'results'> & { results: T[] }
/** Every run of `suite` in the edition as one: the first run's protocol, each model's newest row. */
function merged<T extends { model: string }>(suite: string): RunOf<T & { ollaya: string }> | undefined {
  const rs = editionRuns.filter((r) => r.suite === suite)
  if (!rs.length) return undefined
  const byModel = new Map<string, T & { ollaya: string }>()
  for (const r of rs) for (const x of r.results as T[]) byModel.set(x.model, { ...x, ollaya: versionOf(r) })
  return { ...rs[0]!, results: [...byModel.values()] }
}
export const publicRun = merged<BenchPublic>('public-benchmark')
export const typedRun = merged<TypedResult>('typed-decisions')

/** Rank by the score's interval: one plus the number of models whose whole interval is above this one's. */
function ranked(results: ScoreResult[]): (ScoreResult & { rank: number })[] {
  return results.map((r) => ({ ...r, rank: 1 + results.filter((o) => o.score_ci[0] > r.score_ci[1]).length }))
}
const scores = new Map(ranked(edition?.score.results ?? []).map((r) => [r.model, r]))

function runtimeOf(model: Model | undefined, tag: string): Row['runtime'] {
  const t = model?.tags.find((x) => x.name === tag)
  if (!t || t.kind === 'router') return 'router'
  return t.registry.config.format === 'gguf' ? 'llama.cpp' : 'ONNX Runtime'
}

/** The parity gates a model passed on its way into the library: the runtime against its authors' reference, as cited in the docs. */
function parityFor(name: string): ParityResult[] {
  // The compiled docs first, then the recorded runs by date: the newest row per device wins.
  const ordered = [...parityRuns].sort((a, b) => (a.date ?? '').localeCompare(b.date ?? ''))
  const out = new Map<string, ParityResult>()
  for (const run of ordered)
    for (const r of run.results)
      if (shortName(r.model) === name && r.kind === 'runtime' && (r.cite || r.gate))
        out.set(r.device, r.cite ? r : { ...r, cite: `results/runs/${run.file}`, device_detail: `${machines[run.machine ?? '']?.headline ?? run.machine} ${r.device === 'cpu' ? 'CPU' : r.device.toUpperCase()}` })
  return [...out.values()]
}

function rowFor(name: string, variants: Row[] = []): Row {
  const [family, tag = 'latest'] = name.split(':') as [string, string | undefined]
  const model = catalog.find((m) => m.name === family)
  const t = model?.tags.find((x) => x.name === tag)
  const speed = new Map<string, SpeedResult>()
  for (const d of devices) {
    const s = d.results.get(name)
    if (s) speed.set(d.slug, s)
  }
  const pub = publicRun?.results.find((r) => r.model === name)
  const typed = typedRun?.results.find((r) => r.model === name)
  return {
    name,
    family,
    model,
    title: model?.title ?? family,
    publisher: model?.publisher?.name ?? null,
    description: t?.summary ?? model?.description ?? '',
    params: t?.params ?? t?.registry.config.parameterSize ?? null,
    runtime: runtimeOf(model, tag),
    license: t?.registry.config.license ?? model?.license ?? '',
    digest: pub?.digest ?? typed?.digest ?? [...speed.values()].find((s) => s.digest)?.digest ?? t?.registry.digest ?? null,
    score: scores.get(name),
    public: pub,
    typed,
    speed,
    parity: parityFor(name),
    variants,
    builtin: !!t?.builtinQuestions,
    router: t?.kind === 'router',
  }
}

/** Whether the edition measured a tag at all (a variant is measured without being in `models`). */
const measured = (name: string) => devices.some((d) => d.results.has(name)) || !!typedRun?.results.some((r) => r.model === name)

export const rows: Row[] = (edition?.models ?? []).map((name) =>
  rowFor(name, measured(`${name}-fp16`) && published(`${name}-fp16`) ? [rowFor(`${name}-fp16`)] : []),
)

/** Share of the 5,880 questions the model rejected (an option count, a score scale or an input length it can't take). */
export function rejectedShare(r: Row): number {
  if (!r.public || !r.typed) return 0
  return (r.public.errors + r.typed.errors * 5) / (r.public.records + r.typed.questions)
}

export const byScore = (a: Row, b: Row) => (b.score?.score ?? -1) - (a.score?.score ?? -1) || a.name.localeCompare(b.name)
export const scoredRows = rows.filter((r) => r.score).sort(byScore)

/** Rows with a score and a time on `d`: the points of its accuracy-against-latency chart. */
export function pointsOn(d: Device) {
  return scoredRows.filter((r) => r.speed.get(d.slug)?.p50_ms !== undefined)
}

/** The models nothing beats on both score and latency on `d`. */
export function frontier(d: Device): Set<string> {
  const pts = pointsOn(d).sort((a, b) => a.speed.get(d.slug)!.p50_ms! - b.speed.get(d.slug)!.p50_ms! || b.score!.score - a.score!.score)
  const out = new Set<string>()
  let best = -Infinity
  for (const r of pts) {
    if (r.score!.score > best) {
      out.add(r.name)
      best = r.score!.score
    }
  }
  return out
}

/** The 14 tasks of the score, in display order, with their labels. */
export const TASKS: { id: string; label: string; area: string }[] = [
  { id: 'typed-decisions', label: 'typed-decisions', area: 'Workflows' },
  { id: 'massive-en-US', label: 'MASSIVE en', area: 'Classification' },
  { id: 'massive-de-DE', label: 'MASSIVE de', area: 'Classification' },
  { id: 'civil_comments', label: 'Civil Comments', area: 'Classification' },
  { id: 'aegis2', label: 'Aegis 2', area: 'Classification' },
  { id: 'multinli', label: 'MultiNLI', area: 'Understanding' },
  { id: 'vitaminc-dev', label: 'VitaminC', area: 'Understanding' },
  { id: 'paws', label: 'PAWS', area: 'Understanding' },
  { id: 'boolq', label: 'BoolQ', area: 'Knowledge' },
  { id: 'squad2', label: 'SQuAD 2', area: 'Knowledge' },
  { id: 'pubmedqa', label: 'PubMedQA', area: 'Knowledge' },
  { id: 'helpsteer2', label: 'HelpSteer 2', area: 'Judging' },
  { id: 'summeval-relevance', label: 'SummEval rel.', area: 'Judging' },
  { id: 'summeval-consistency', label: 'SummEval cons.', area: 'Judging' },
]

/** URL path of a model's page; the colon stays (public/_redirects in the bench build serves it). */
export const modelUrl = (name: string) => `/models/${name}`
/** Files a model's page is written to: the colon file for plain static hosts, and the rewrite target. */
export function modelFiles(name: string): string[] {
  const [family, tag] = name.split(':')
  return tag ? [`models/${name}.html`, `models/${family}/${tag}.html`] : [`models/${name}.html`]
}

/** A model's badge: /badge/<family>/<tag>.svg, no colon in the path (Cloudflare would 307 it). */
export const badgeUrl = (name: string) => `/badge/${name.replace(':', '/')}.svg`

export const deviceUrl = (d: Device) => (d === devices[0] ? '/' : `/hardware/${d.slug}`)

export function fmtMs(ms: number | undefined): string {
  if (ms === undefined) return '–'
  if (ms >= 10000) return `${(ms / 1000).toFixed(1)} s`
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`
  if (ms >= 100) return `${Math.round(ms)} ms`
  return `${ms.toFixed(1)} ms`
}

export function fmtMb(mb: number | undefined): string {
  if (mb === undefined) return '–'
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${Math.round(mb)} MB`
}

/** The memory that matters on a device: the GPU's for a GPU, the runner's peak resident memory on a CPU. */
export function memoryOn(d: Device, s: SpeedResult | undefined): number | undefined {
  if (!s) return undefined
  return d.device === 'cpu' ? s.ram_peak_mb : s.vram_mb
}
