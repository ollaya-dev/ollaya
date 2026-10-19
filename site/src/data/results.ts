/**
 * The measurements behind /results, read from ../results by scripts/gen-results.mjs. Each run is
 * one file there, written by the tool that measured it (bench_latency.py, bench_public.py, the
 * parity examples); nothing on the page is typed in by hand.
 */
import { registryModels } from '../generated/registry'
import { machinesRaw, runsRaw } from '../generated/results'

export interface Machine {
  label: string
  /** The part a reader recognises it by: its GPU, or the chip. */
  headline?: string
  /** Our own test bench, or a community measurement. */
  kind?: 'ours' | 'community'
  /** What we measure on it. */
  uses?: string[]
  cpu: string
  gpus: string[]
  ram_gb?: number
  os: string
  driver?: string
  /** A community measurement: the GitHub user who ran it, and where it was reported. */
  by?: string
  source?: string
}

export interface PublicResult {
  server: 'ollaya' | 'ollama'
  model: string
  /** Ollama rows: the quantization and size `ollama pull` serves. */
  weights?: string | null
  records: number
  errors: number
  macro_accuracy: number
  micro_accuracy: number
  macro_ece: number
  pooled_ece: number
  brier: number
  median_ms: number
  p95_ms: number
  by_type: Record<string, number>
  per_subset: Record<string, { n: number; errors: number; accuracy: number; ece: number | null }>
}

export interface LatencyResult {
  model: string
  questions?: string
  p50_ms?: number
  p90_ms?: number
  min_ms?: number
  device?: string
  precision?: string
  load_ms?: number
  error?: string
}

export interface ParityResult {
  model: string
  family?: string
  device: string
  device_detail?: string
  gpu?: string
  cpu?: string
  kind?: 'runtime' | 'export'
  questions?: number | null
  decisions?: string | null
  decisions_same?: number
  max_logit_diff?: number | null
  max_prob_diff?: number | null
  p50_ms?: number
  pass?: boolean | null
  date?: string | null
  cite?: string
  reference_note?: string
  /** Rows written by convert/ollaya_convert/parity_run.py: the gate the tag passed and the export it checked. */
  gate?: 'fp32' | 'fp16'
  p99_prob_diff?: number
  export_sha256?: string
}

export interface CrossDevice {
  model: string
  device: string
  questions: number
  decisions_same: number
  max_logprob_diff: number
}

interface RunBase {
  file: string
  schema: 1
  date?: string
  machine?: string
  device?: string
  note?: string
  /** The Ollaya Bench edition a run belongs to (bench_edition.py). */
  edition?: string
}
export interface PublicRun extends RunBase {
  suite: 'public-benchmark'
  servers: Record<string, string>
  results: PublicResult[]
}
export interface LatencyRun extends RunBase {
  suite: 'latency-triage'
  ollaya: string
  results: LatencyResult[]
}
export interface ParityRun extends RunBase {
  suite: 'parity'
  results: ParityResult[]
  cross_device?: { note: string; results: CrossDevice[] }
}
export type Run = PublicRun | LatencyRun | ParityRun

export const machines = machinesRaw as Record<string, Machine>
export const runs = runsRaw as Run[]

export const publicRuns = runs.filter((r): r is PublicRun => r.suite === 'public-benchmark')
export const latencyRuns = runs.filter((r): r is LatencyRun => r.suite === 'latency-triage')
export const parityRuns = runs.filter((r): r is ParityRun => r.suite === 'parity')

/**
 * The comparison with Ollama that this page and the home page draw: the newest public-benchmark run of Ollama,
 * beside the newest Ollaya row of each model that the Ollaya Bench runs (an edition and the releases that updated
 * it, ADR 0009) measured on the same machine. A run that measured both servers stands as it is.
 */
export const publicBench: PublicRun | undefined = (() => {
  const ollama = publicRuns.filter((r) => 'ollama' in r.servers).at(-1)
  if (!ollama || 'ollaya' in ollama.servers) return ollama
  const ollaya = new Map<string, PublicResult>()
  let version = ''
  for (const r of publicRuns)
    if (r.machine === ollama.machine && r.edition && 'ollaya' in r.servers)
      for (const x of r.results) {
        ollaya.set(shortName(x.model), x)
        version = r.servers.ollaya!
      }
  return { ...ollama, servers: { ...ollama.servers, ollaya: version }, results: [...ollama.results, ...ollaya.values()] }
})()

/**
 * The name a run's model goes by on the page: without a registry host (`127.0.0.1:18742/library/
 * jeb:4b` is `jeb:4b`), and `:latest` as the tag it points to (`nli:latest` is
 * `nli:deberta-v3-large`), so runs that pulled different aliases line up.
 */
export function shortName(model: string): string {
  const name = model.replace(/^.*\/library\//, '')
  const [base, tag] = name.split(':') as [string, string | undefined]
  if (tag && tag !== 'latest') return name
  const m = registryModels.find((r) => r.name === base)
  const latest = m?.tags.find((t) => t.name === 'latest')
  const same = latest && m?.tags.find((t) => t.name !== 'latest' && t.digest === latest.digest)
  return same ? `${base}:${same.name}` : name
}

/** A device column of the speed chart: one machine and what it ran on. */
export interface DeviceSeries {
  key: string
  machine: string
  device: string
  label: string
  detail: string
  /** model → median of five-question requests, in ms. */
  p50: Map<string, number>
}

/**
 * Every (machine, device) pair the latency runs cover, GPUs first. Runs of the same pair (one
 * per model store) merge; a model timed twice keeps the newer run's number.
 */
export function latencySeries(): DeviceSeries[] {
  const byKey = new Map<string, DeviceSeries>()
  for (const run of latencyRuns) {
    const key = `${run.machine}/${run.device}`
    const m = machines[run.machine ?? '']
    const gpu = m?.gpus[0]?.replace(/^NVIDIA GeForce /, '').replace(/, .*$/, '') ?? 'GPU'
    const cpu = m?.cpu.replace(/ \(.*$/, '') ?? 'CPU'
    let s = byKey.get(key)
    if (!s) {
      s = {
        key,
        machine: run.machine ?? '',
        device: run.device ?? '',
        label: run.device === 'cpu' ? cpu.replace(/^(AMD Ryzen|Intel Core) /, '') : gpu,
        detail: run.device === 'cpu' ? 'CPU' : (run.device ?? '').toUpperCase(),
        p50: new Map(),
      }
      byKey.set(key, s)
    }
    for (const r of run.results) if (r.p50_ms !== undefined && !r.error) s.p50.set(shortName(r.model), r.p50_ms)
  }
  // GPUs before CPUs; within each, the machines in machines.json order.
  const order = Object.keys(machines)
  return [...byKey.values()].sort(
    (a, b) => Number(a.device === 'cpu') - Number(b.device === 'cpu') || order.indexOf(a.machine) - order.indexOf(b.machine) || a.key.localeCompare(b.key),
  )
}
