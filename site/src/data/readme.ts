/**
 * The README's measured numbers, written between `<!-- results:<name> -->` and `<!-- /results:<name> -->`.
 * `npm run build` fails when README.md differs from these; `npm run readme` rewrites the blocks.
 */
import { benchEdition, result, rowsOf, value } from './numbers'

const time = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`)
const code = (tag: string) => `\`${tag}\``
const host = (origin: string) => origin.replace(/^https?:\/\//, '')
const list = (xs: string[]) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`)

function range(device: 'gpu' | 'cpu'): string {
  const ms = rowsOf(device)
    .map(([, r]) => r.p50_ms)
    .filter((v): v is number => typeof v === 'number')
    .sort((a, b) => a - b)
  return `${time(ms[0]!)} to ${time(ms.at(-1)!)}`
}

function bench(origin: string): string {
  const { id, ollaya, updates, machine } = benchEdition
  const gpu = machine?.headline ?? ''
  const cpu = (machine?.cpu ?? '').replace(/^AMD Ryzen |\s*\(.*\)$/g, '')
  const top = rowsOf('score')
    .map(([tag, r]) => ({ tag, score: r.score as number, trained: ((r.trained_on as string[] | undefined) ?? []).includes('typed-decisions') }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
  const fastest = top.reduce((a, b) => (value(a.tag, 'gpu.p50_ms') <= value(b.tag, 'gpu.p50_ms') ? a : b))
  const ece = new Map<string, string[]>()
  for (const [tag, r] of rowsOf('public').sort((a, b) => (a[1].pooled_ece as number) - (b[1].pooled_ece as number)))
    if (typeof r.pooled_ece === 'number') ece.set(r.pooled_ece.toFixed(3), [...(ece.get(r.pooled_ece.toFixed(3)) ?? []), tag])
  const calibration = [...ece].slice(0, 3).map(([v, tags]) => `${v} (${tags.map(code).join(', ')})`)
  return [
    `Edition ${id} ran Ollaya ${ollaya} on an ${machine?.label ?? gpu}${updates.length ? `, and ${list(updates)} added what ${updates.length > 1 ? 'they' : 'it'} changed` : ''}:`,
    '',
    `- **Score.** Chance-corrected accuracy over 14 tasks, with 95 % intervals: ${list(
      top.map((t) => `${code(t.tag)} ${t.score.toFixed(1)}${t.trained ? ' (scored without typed-decisions, which it was trained on)' : ''}`),
    )}. The fastest of these, ${code(fastest.tag)}, answers five questions in ${result(fastest.tag, 'gpu.p50_ms', 0)} ms on the ${gpu}.`,
    `- **Calibration.** The lowest calibration errors on the public benchmark are ${list(calibration)}.`,
    `- **Speed and memory.** Five-question requests through the HTTP API take ${range('gpu')} on the ${gpu} and ${range('cpu')} on its ${cpu}. Every model's VRAM and RAM are on its page.`,
    '- **Parity.** Before a model ships, its runtime is checked question by question against the authors\' code (or llama.cpp\'s own server, for GGUF models) on each device it runs on.',
    '',
    `Earlier runs, the comparison with Ollama on the same GPU (on the same Nimble weights the calibration error is ${result('nimble:9b', 'public.pooled_ece', 3)} on Ollaya and ${result('ollama/nimble', 'public.pooled_ece', 3)} on Ollama) and the Windows, CUDA and Vulkan measurements are at [${host(origin)}/results](${origin}/results).`,
  ].join('\n')
}

const recommended = (origin: string) =>
  `${code('winnow:e4b')} is the recommended model: ${result('winnow:e4b', 'typed.accuracy', 3)} accuracy on typed decisions (TypeSafe reports 0.738 for its Jev) and ${result('winnow:e4b', 'gpu.p50_ms', 0)} ms for five questions on an ${benchEdition.machine?.headline}. It is a 4B-class language model, so without an NVIDIA GPU start with ${code('laya')}: ${code('laya:multilingual')} answers five questions in ${result('laya:multilingual', 'cpu.p50_ms', 0)} ms on a CPU. All models and their numbers: [${host(origin)}/search](${origin}/search).`

export const readmeBlocks = (origin: string): Record<string, string> => ({ recommended: recommended(origin), bench: bench(origin) })

/** README.md with every results block replaced, linking to `origin` (the production site); throws when a block is missing. */
export function withReadmeBlocks(readme: string, origin: string): string {
  let out = readme
  for (const [name, body] of Object.entries(readmeBlocks(origin))) {
    const re = new RegExp(`(<!-- results:${name} -->)[\\s\\S]*?(<!-- /results:${name} -->)`)
    if (!re.test(out)) throw new Error(`README.md has no <!-- results:${name} --> block`)
    out = out.replace(re, (_, a: string, b: string) => `${a}\n${body}\n${b}`)
  }
  return out
}
