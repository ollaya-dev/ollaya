/**
 * The home page's terminal example, read from hero.json: the real output of
 * `ollaya run <model> --preset <preset> --format json --verbose "<message>"`, recorded by
 * scripts/record-hero.mjs. Nothing in the terminal is typed in by hand; a new model or message is a
 * new recording, and an answer type this file does not know fails the build.
 */
import fixture from './hero.json'

export interface HeroOption {
  label: string
  /** The option's whole wording, where `label` shortens it. */
  full?: string
  /** The probability the model gave this option. */
  p: number
}

/** A question: its answer, the most probable of its own options, and the others, most probable first. */
export interface HeroQuestion {
  name: string
  answer: HeroOption
  others: HeroOption[]
}

const fail = (question: string, why: string): never => {
  throw new Error(`src/data/hero.json: ${question}: ${why}`)
}

const isProbability = (v: unknown): v is number => typeof v === 'number' && v >= 0 && v <= 1

function probabilities(question: string, v: unknown): [string, number][] {
  const entries = typeof v === 'object' && v !== null ? Object.entries(v) : []
  if (!entries.length) fail(question, 'no probabilities (record with --verbose)')
  return entries.map(([k, p]) => (isProbability(p) ? [k, p] : fail(question, `bad probability for ${k}`)))
}

// A score's level goes by the first alternative of its legend ("very angry or using strong
// language" is "very angry"), which fits a phone; the whole legend stays in the title and for
// screen readers.
const shortLevel = (legend: string) => legend.split(' or ')[0] || legend

function questionOf(question: string, a: Record<string, unknown>): HeroQuestion {
  let options: HeroOption[]
  if (a.type === 'choice') {
    options = probabilities(question, a.probabilities).map(([label, p]) => ({ label, p }))
  } else if (a.type === 'noul') {
    const yes = isProbability(a.noul) ? a.noul : fail(question, 'bad noul')
    options = [
      { label: 'yes', p: yes },
      { label: 'no', p: 1 - yes },
    ]
  } else if (a.type === 'score') {
    const legend = typeof a.legend === 'object' && a.legend !== null ? (a.legend as Record<string, unknown>) : {}
    options = probabilities(question, a.probabilities).map(([level, p]) => {
      const full = legend[level]
      if (typeof full !== 'string') return fail(question, `no legend for level ${level}`)
      const label = shortLevel(full)
      return label === full ? { label, p } : { label, full, p }
    })
  } else {
    return fail(question, `the terminal cannot show a "${String(a.type)}" answer yet`)
  }
  options.sort((x, y) => y.p - x.p)
  const [answer, ...others] = options
  if (!answer) return fail(question, 'no options')
  if (a.type === 'choice' && answer.label !== a.choice) fail(question, `the answer ${String(a.choice)} is not the top option`)
  return { name: question, answer, others }
}

/** A recording in hero.json's shape (scripts/record-hero.mjs writes them), as the pages draw it. */
export function heroOf(f: typeof fixture) {
  return {
    model: f.model,
    preset: f.preset,
    message: f.message,
    device: f.device,
    latencyMs: f.latency_ms,
    questions: Object.entries(f.output.answers).map(([name, a]) => questionOf(name, a as Record<string, unknown>)),
  }
}

export const hero = heroOf(fixture)
