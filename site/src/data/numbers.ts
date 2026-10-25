/**
 * Every measured number on the site comes from ../results (docs/decisions/0009-measurement-per-release.md):
 *
 *   result('kev:4b', 'typed.accuracy', 3)            in code
 *   {{result kev:4b typed.accuracy 3}}               in Markdown (library pages, docs), filled in by withResults
 *
 * Paths, per tag, from the current Ollaya Bench edition and the releases that updated it (a later row wins):
 *   score, score_ci.0, score_ci.1     the edition score
 *   tasks.<task>                      one task's chance-corrected score in it (tasks.typed-decisions, …)
 *   typed.<field>                     typed-decisions (accuracy, ece, median_ms, …)
 *   public.<field>                    Bespoke Labs' public benchmark (macro_accuracy, pooled_ece, median_ms, …)
 *   gpu.<field>, cpu.<field>          latency-triage on the edition's machine (p50_ms, vram_mb, ram_peak_mb, load_ms, …)
 *   parity.<device>.<field>           the newest runtime parity row (max_logit_diff, max_prob_diff, decisions_same, …)
 * A tag written `ollama/<tag>` reads Ollama's newest runs instead (typed and public). `digits` fixes the decimals;
 * `e<digits>` writes an exponent instead (`e1`: 5.8e-6).
 * An unknown tag or path throws, so the build fails rather than print a missing number.
 */
import { editionsRaw } from '../generated/results'
import { machines, runs, shortName } from './results'

type Row = Record<string, unknown> & { model: string }
type AnyRun = { file: string; suite: string; device?: string; machine?: string; servers?: Record<string, string>; edition?: string; results: Row[] }
type Edition = { id: string; machine: string; runs: string[]; score: { results: Row[] } }

const edition = (editionsRaw as Edition[]).at(-1)
const all = runs as unknown as AnyRun[]
const inEdition = all.filter((r) => edition?.runs.includes(r.file))

/** tag -> the newest row of the runs that `pick` selects, in file (date) order. */
function newest(pick: (r: AnyRun) => boolean, server?: string): Map<string, Row> {
  const out = new Map<string, Row>()
  for (const r of all.filter(pick))
    // Ollama's names stay as Ollama spells them: `nimble` there is not Ollaya's `nimble:latest`.
    for (const x of r.results) if (!server || (x.server ?? 'ollaya') === server) out.set(server === 'ollama' ? x.model : shortName(x.model), x)
  return out
}

const tables: Record<string, Map<string, Row>> = {
  score: new Map((edition?.score.results ?? []).map((r) => [r.model, r])),
  typed: newest((r) => inEdition.includes(r) && r.suite === 'typed-decisions'),
  public: newest((r) => inEdition.includes(r) && r.suite === 'public-benchmark', 'ollaya'),
  gpu: newest((r) => inEdition.includes(r) && r.suite === 'latency-triage' && r.device === 'cuda'),
  cpu: newest((r) => inEdition.includes(r) && r.suite === 'latency-triage' && r.device === 'cpu'),
  'ollama/typed': newest((r) => r.suite === 'typed-decisions' && !!r.servers && 'ollama' in r.servers, 'ollama'),
  'ollama/public': newest((r) => r.suite === 'public-benchmark' && !!r.servers && 'ollama' in r.servers, 'ollama'),
}
const parity = new Map<string, Row>()
for (const r of all.filter((r) => r.suite === 'parity').sort((a, b) => String((a as { date?: string }).date ?? '').localeCompare(String((b as { date?: string }).date ?? ''))))
  for (const x of r.results) if (x.kind === 'runtime') parity.set(`${shortName(x.model)}/${x.device}`, x)

/** The raw value at `path` for `tag`. */
export function value(tag: string, path: string): number {
  const [head, ...rest] = path.split('.')
  let row: Row | undefined
  let keys = rest
  if (head === 'parity') {
    row = parity.get(`${tag}/${rest[0]}`)
    keys = rest.slice(1)
  } else if (head === 'score' || head === 'score_ci' || head === 'tasks') {
    row = tables.score!.get(tag)
    keys = [head, ...rest]
  } else {
    const table = tag.startsWith('ollama/') ? tables[`ollama/${head}`] : tables[head!]
    if (!table) throw new Error(`result: unknown path ${path}`)
    row = table.get(tag.replace(/^ollama\//, ''))
  }
  if (!row) throw new Error(`result: no ${head} row for ${tag} in results/`)
  const v = keys.reduce<unknown>((o, k) => (o as Record<string, unknown> | undefined)?.[k], row)
  if (typeof v !== 'number') throw new Error(`result: ${tag} ${path} is not a number in results/`)
  return v
}

/** The value at `path` for `tag`, as the page prints it. */
export function result(tag: string, path: string, digits?: number): string {
  const v = value(tag, path)
  return digits === undefined ? String(v) : v.toFixed(digits)
}

/** Fills in the {{result <tag> <path> [digits]}} references of rendered Markdown. */
export const withResults = (html: string) =>
  html.replace(/\{\{result ([^\s}]+) ([^\s}]+)(?: (e?)(\d+))?\}\}/g, (_, tag: string, path: string, e?: string, d?: string) =>
    e ? value(tag, path).toExponential(Number(d)) : result(tag, path, d === undefined ? undefined : Number(d)),
  )

/** The value, or undefined when results/ has no such row (a model the suite does not apply to). */
export function measured(tag: string, path: string): number | undefined {
  try {
    return value(tag, path)
  } catch {
    return undefined
  }
}

/** Whether the tag was trained on typed-decisions' train split (its typed number is not a held-out one). */
export const trainedOnTyped = (tag: string) => tables.typed!.get(tag)?.trained_on_split === true

/** The GPU the edition times models on, as pages name it ("RTX 5090"). */
export const benchGpu = machines[edition?.machine ?? '']?.headline ?? ''

/** The [tag, row] pairs of one table (score, typed, public, gpu, cpu), for pages that rank or range over them. */
export const rowsOf = (table: 'score' | 'typed' | 'public' | 'gpu' | 'cpu') => [...tables[table]!.entries()]

/** The current edition: its id, the release that ran it, the releases that updated it and its machine. */
export const benchEdition = {
  id: edition?.id ?? '',
  ollaya: (edition as { ollaya?: string } | undefined)?.ollaya ?? '',
  updates: ((edition as { updates?: { release: string }[] } | undefined)?.updates ?? []).map((u) => u.release),
  machine: machines[edition?.machine ?? ''],
}
