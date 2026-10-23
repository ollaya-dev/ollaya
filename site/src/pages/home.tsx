/** The home page: the hero with its recorded examples, the numbers, the case, the comparison, the API, the models, the platforms. */
import type { Child } from 'hono/jsx'
import { Icon, type IconName } from '../components/Icon'
import { LogoMark } from '../components/Logo'
import { CopyButton } from '../components/ui'
import { highlight } from '../lib/highlight'
import { catalog } from '../data/catalog'
import { benchGpu, result } from '../data/numbers'
import stats from '../data/stats.json'
import { asset, hasAsset } from '../lib/assets'
import { benchUrl } from '../lib/bench'
import { githubStars } from '../lib/stars'
import { GITHUB_URL, LOCAL_API } from '../site'
import { hero, heroOf, type HeroQuestion } from '../data/hero'
import heroExamples from '../data/hero-examples.json'
import triage from '../../../crates/ollaya-api/src/presets/triage.json'
import { publicBench, shortName } from '../data/results'

const lastWeek = stats.weeks.at(-1)!
const fmt = (n: number) => n.toLocaleString('en-US')

/**
 * Publishers worth a mark on the page. The SVG paths are simple-icons' (CC0) drawings of each brand's
 * own mark; the PNGs are the organisation's own Hugging Face avatar. Trademarks stay their owners'.
 */
const PUBLISHERS: { name: string; path?: string; png?: string }[] = [
  { name: 'Perplexity AI', path: 'M22.3977 7.0896h-2.3106V.0676l-7.5094 6.3542V.1577h-1.1554v6.1966L4.4904 0v7.0896H1.6023v10.3976h2.8882V24l6.932-6.3591v6.2005h1.1554v-6.0469l6.9318 6.1807v-6.4879h2.8882V7.0896zm-3.4657-4.531v4.531h-5.355l5.355-4.531zm-13.2862.0676 4.8691 4.4634H5.6458V2.6262zM2.7576 16.332V8.245h7.8476l-6.1149 6.1147v1.9723H2.7576zm2.8882 5.0404v-3.8852h.0001v-2.6488l5.7763-5.7764v7.0111l-5.7764 5.2993zm12.7086.0248-5.7766-5.1509V9.0618l5.7766 5.7766v6.5588zm2.8882-5.0652h-1.733v-1.9723L13.3948 8.245h7.8478v8.087z' },
  { name: 'Cloudflare', path: 'M16.5088 16.8447c.1475-.5068.0908-.9707-.1553-1.3154-.2246-.3164-.6045-.499-1.0615-.5205l-8.6592-.1123a.1559.1559 0 0 1-.1333-.0713c-.0283-.042-.0351-.0986-.021-.1553.0278-.084.1123-.1484.2036-.1562l8.7359-.1123c1.0351-.0489 2.1601-.8868 2.5537-1.9136l.499-1.3013c.0215-.0561.0293-.1128.0147-.168-.5625-2.5463-2.835-4.4453-5.5499-4.4453-2.5039 0-4.6284 1.6177-5.3876 3.8614-.4927-.3658-1.1187-.5625-1.794-.499-1.2026.119-2.1665 1.083-2.2861 2.2856-.0283.31-.0069.6128.0635.894C1.5683 13.171 0 14.7754 0 16.752c0 .1748.0142.3515.0352.5273.0141.083.0844.1475.1689.1475h15.9814c.0909 0 .1758-.0645.2032-.1553l.12-.4268zm2.7568-5.5634c-.0771 0-.1611 0-.2383.0112-.0566 0-.1054.0415-.127.0976l-.3378 1.1744c-.1475.5068-.0918.9707.1543 1.3164.2256.3164.6055.498 1.0625.5195l1.8437.1133c.0557 0 .1055.0263.1329.0703.0283.043.0351.1074.0214.1562-.0283.084-.1132.1485-.204.1553l-1.921.1123c-1.041.0488-2.1582.8867-2.5527 1.914l-.1406.3585c-.0283.0713.0215.1416.0986.1416h6.5977c.0771 0 .1474-.0489.169-.126.1122-.4082.1757-.837.1757-1.2803 0-2.6025-2.125-4.727-4.7344-4.727' },
  { name: 'PostHog', path: 'M9.854 14.5 5 9.647.854 5.5A.5.5 0 0 0 0 5.854V8.44a.5.5 0 0 0 .146.353L5 13.647l.147.146L9.854 18.5l.146.147v-.049c.065.03.134.049.207.049h2.586a.5.5 0 0 0 .353-.854L9.854 14.5zm0-5-4-4a.487.487 0 0 0-.409-.144.515.515 0 0 0-.356.21.493.493 0 0 0-.089.288V8.44a.5.5 0 0 0 .147.353l9 9a.5.5 0 0 0 .853-.354v-2.585a.5.5 0 0 0-.146-.354l-5-5zm1-4a.5.5 0 0 0-.854.354V8.44a.5.5 0 0 0 .147.353l4 4a.5.5 0 0 0 .853-.354V9.854a.5.5 0 0 0-.146-.354l-4-4zm12.647 11.515a3.863 3.863 0 0 1-2.232-1.1l-4.708-4.707a.5.5 0 0 0-.854.354v6.585a.5.5 0 0 0 .5.5H23.5a.5.5 0 0 0 .5-.5v-.6c0-.276-.225-.497-.499-.532zm-5.394.032a.8.8 0 1 1 0-1.6.8.8 0 0 1 0 1.6zM.854 15.5a.5.5 0 0 0-.854.354v2.293a.5.5 0 0 0 .5.5h2.293c.222 0 .39-.135.462-.309a.493.493 0 0 0-.109-.545L.854 15.501zM5 14.647.854 10.5a.5.5 0 0 0-.854.353v2.586a.5.5 0 0 0 .146.353L4.854 18.5l.146.147h2.793a.5.5 0 0 0 .353-.854L5 14.647z' },
  { name: 'Bespoke Labs', png: 'bespokelabs' },
  { name: 'Qwen', path: 'M23.919 14.545 20.817 9.17l1.47-2.544a.56.56 0 0 0 0-.566l-1.633-2.83a.57.57 0 0 0-.49-.283h-6.207L12.487.402a.57.57 0 0 0-.49-.284H8.732a.56.56 0 0 0-.49.284L5.139 5.775h-2.94a.56.56 0 0 0-.49.284L.077 8.887a.56.56 0 0 0 0 .567L3.18 14.83l-1.47 2.545a.56.56 0 0 0 0 .566l1.634 2.83a.57.57 0 0 0 .49.283h6.205l1.47 2.545a.57.57 0 0 0 .49.284h3.266a.57.57 0 0 0 .49-.284l3.104-5.375h2.94a.57.57 0 0 0 .49-.283l1.634-2.828a.55.55 0 0 0-.004-.568M8.733.686l1.634 2.828-1.634 2.828H21.8L20.164 9.17H7.425L5.63 6.06Zm1.306 19.801-6.205-.002 1.634-2.83h3.265L2.201 6.344h3.267q3.182 5.517 6.367 11.032zm10.124-5.66L18.53 12l-6.532 11.315-1.634-2.83c2.129-3.673 4.25-7.351 6.373-11.028h3.592l3.102 5.374z' },
  { name: 'Strands Agents', png: 'strandsagents' },
  { name: 'Knowledgator', png: 'knowledgator' },
  { name: 'Convai Innovations', png: 'convai' },
]

function Mark({ p }: { p: (typeof PUBLISHERS)[number] }) {
  if (p.path)
    return (
      <svg viewBox="0 0 24 24" class="size-9 shrink-0" fill="currentColor" aria-hidden="true">
        <path d={p.path} />
      </svg>
    )
  return <img src={asset(`logos/${p.png}.png`)} class="size-9 shrink-0 rounded-[4px] brightness-150 grayscale" alt="" width="36" height="36" loading="lazy" decoding="async" />
}

export const JEV = { acc: 0.738, text: '236–276 ms' }
// Bespoke Labs' public decision benchmark (github.com/bespokelabsai/nimble, docs/PUBLIC_BENCHMARKS.md):
// 13 subsets of 11 human-labeled datasets, 3,880 questions, rebuilt byte for byte from Bespoke's manifests
// and scored with Bespoke's own runner (convert/ollaya_convert/bench_public.py). Both servers ran on the same
// RTX 5090, one request at a time through /v1/systemone. Accuracy: macro average over the 13 subsets. ECE:
// ten-bin expected calibration error of the top probability, pooled over the questions (lower is better).
// Latency: median request, HTTP included. The numbers are results/: the Ollaya Bench edition and its updates for
// Ollaya, Ollama 0.40.6's runs for Ollama (publicBench); docs/measurements/2026-10-ollama-0.40.md discusses them.
type VsRun = { tag: string; server: 'ollaya' | 'ollama'; acc: number; ece: number; ms: number }
const VS_TAGS: [string, VsRun['server']][] = [
  ['pplx-decider:flash', 'ollaya'], ['pplx-decider', 'ollama'], ['jeb:27b', 'ollaya'], ['winnow:12b', 'ollaya'],
  ['clef-flash', 'ollama'], ['clef:flash', 'ollaya'], ['jeb:9b', 'ollaya'], ['nimble', 'ollama'], ['tev1', 'ollama'],
  ['nimble:9b', 'ollaya'], ['kev:4b', 'ollaya'], ['kev:4b-q8_0', 'ollama'], ['tev1:0.8b', 'ollama'],
  ['laya:multilingual', 'ollaya'],
]
export const vsRuns: VsRun[] = VS_TAGS.map(([tag, server]) => {
  // Ollama's names stay Ollama's: its `pplx-decider` is the 27B, not Ollaya's `pplx-decider:latest`.
  const r = publicBench?.results.find((x) => x.server === server && (server === 'ollama' ? x.model : shortName(x.model)) === tag)
  if (!r) throw new Error(`home: no ${server} public-benchmark row for ${tag} in results/`)
  return { tag, server, acc: r.macro_accuracy, ece: r.pooled_ece, ms: Math.round(r.median_ms) }
}).sort((a, b) => b.acc - a.acc)
// The section's copy says Ollaya runs the most accurate model; fail the build if results/ stop saying so.
if (vsRuns[0]!.server !== 'ollaya') throw new Error(`home: ${vsRuns[0]!.tag} on Ollama is now the most accurate; update the vs-ollama copy`)
export const vs = (tag: string, server: VsRun['server']) => vsRuns.find((r) => r.tag === tag && r.server === server)!
export const OLLAMA = `Ollama ${publicBench?.servers.ollama ?? ''}`.trim()
/** How many times faster Ollama is on the same weights, as a range over the pairs both serve. */
export function speedup() {
  const r = [['nimble:9b', 'nimble'], ['clef:flash', 'clef-flash'], ['kev:4b', 'kev:4b-q8_0']].map(([a, b]) => vs(a!, 'ollaya').ms / vs(b!, 'ollama').ms)
  const lo = Math.floor(Math.min(...r))
  const hi = Math.round(Math.max(...r))
  return lo === hi ? String(lo) : `${lo} to ${hi}`
}
// A number TypeSafe publishes for its hosted model (the typed-decisions dataset card); not one we measured.
export const JEV_TYPED = '0.738'
// The accuracy axis of the comparison: a dot on a focused range, not a bar from zero, so the
// differences between 0.58 and 0.77 are visible; the ticks print the range.
const VS_MIN = 0.55
const VS_MAX = 0.85
const VS_TICKS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
const vsPos = (a: number) => `${(((Math.min(Math.max(a, VS_MIN), VS_MAX) - VS_MIN) / (VS_MAX - VS_MIN)) * 100).toFixed(2)}%`

export const vsFeatures: { label: string; ollaya: string; ollama: string }[] = [
  { label: 'Decision models', ollaya: `${catalog.length} families`, ollama: '9' },
  { label: 'CPU-fast encoders', ollaya: 'laya, largha, nli, gliclass, von, decima', ollama: 'laya' },
  { label: 'Probabilities', ollaya: "Author's temperature", ollama: 'Raw softmax' },
  { label: 'Options per question', ollaya: '255', ollama: '26 (Kev, Strands: 255)' },
  { label: 'Questions per request', ollaya: '256', ollama: '64' },
  { label: 'TypeSafe endpoints', ollaya: 'systemone, decisions, models', ollama: 'systemone' },
  { label: 'Weights', ollaya: "Author's files, sha256-pinned", ollama: 'Re-hosted GGUF' },
]

const mostAccurate = vs('pplx-decider:flash', 'ollaya')

export const compatRequest = `# Point the TypeSafe SDK at Ollaya
export TYPESAFE_BASE_URL=${LOCAL_API}
export TYPESAFE_API_KEY=local        # any value works
export TYPESAFE_DEFAULT_MODEL=winnow:e4b

# …or call the compatible endpoint directly
curl ${LOCAL_API}/v1/systemone -d '{
    "model": "winnow:e4b",
    "state": "Can I get an invoice for last month?",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
          "invoice": "Needs an invoice or receipt",
          "refund": "Wants money back",
          "other": "Anything else"
        }
      }
    }
  }'`

export const compatResponse = `{
  "model": "winnow:e4b",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "invoice",
      "confidence": 0.9801,
      "probabilities": {
        "invoice": 0.9868,
        "refund": 0.0026,
        "other": 0.0106
      }
    }
  },
  "usage": {
    "input_tokens": 120,
    "output_tokens": 0
  }
}`


function Install({ origin }: { origin: string }) {
  const INSTALL = `curl -fsSL ${origin}/install.sh | sh`
  return (
    <div class="panel panel-strong inline-flex min-h-11 max-w-full items-center gap-3 rounded-md py-1 pr-1 pl-4 font-mono text-[13px] text-fg" data-copy-scope>
      <span class="text-muted select-none" aria-hidden="true">
        $
      </span>
      <pre class="whitespace-pre-wrap break-all">{INSTALL}</pre>
      <CopyButton class="rounded-md" />
    </div>
  )
}

const h2 = 'text-4xl font-semibold leading-[1.05] tracking-[-0.03em] text-fg md:text-5xl'

function Hero({ origin }: { origin: string }) {
  return (
    <section class="relative overflow-hidden" aria-labelledby="rd-hero">
      <div class="field" aria-hidden="true">
        <i class="a"></i>
        <i class="b"></i>
        <i class="c"></i>
      </div>
      <div class="frame relative px-5 pt-32 pb-16 text-center md:pt-40">
        <h1 id="rd-hero" class="mx-auto max-w-4xl text-[2.75rem] font-semibold leading-[1.02] tracking-[-0.04em] text-fg md:text-7xl">
          The decision layer <br class="hidden md:block" />
          for AI agents.
        </h1>
        <p class="mx-auto mt-6 max-w-[38rem] text-lg leading-relaxed text-body md:text-xl">
          Ollaya runs open decision models on your own hardware. Typed questions in, calibrated answers out, in
          milliseconds: routing, guardrails, triage and tool approval without an LLM call.
        </p>
        <div class="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Install origin={origin} />
          <a href={hasAsset('try.js') ? '/try' : '/download'} class="inline-flex h-11 items-center gap-1.5 rounded-md border border-line bg-canvas/40 px-4 text-sm font-medium text-fg hover:border-line-strong">
            {hasAsset('try.js') ? 'Try it in the browser' : 'Download'} <Icon name="arrowRight" class="size-4" />
          </a>
        </div>
        <div class="mx-auto mt-16 max-w-4xl text-left">
          <DecisionDemo />
        </div>
      </div>
    </section>
  )
}

// The hero's example: the real recording in hero.json (a message, the triage preset, the model's
// answers) shown as the request reads to a person: the message, then each question in the preset's
// own words with the answer the model gave and how sure it was.
const QUESTION_TEXT: Record<string, string> = Object.fromEntries(
  Object.entries(triage as Record<string, { instructions: string }>).map(([k, q]) => [k, q.instructions.replace(/`message`/g, 'the message')]),
)
const nice = (label: string) => label.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase())

function AnswerRow({ q, row }: { q: HeroQuestion; row: number }) {
  const runnerUp = q.others[0]
  return (
    <li class="term-row px-5 py-3.5" style={`--row:${row}`}>
      <p class="text-[13px] text-muted">{QUESTION_TEXT[q.name] ?? q.name}</p>
      <div class="mt-1.5 flex items-center gap-4">
        <span class="w-36 shrink-0 truncate text-[15px] font-medium text-fg" title={q.answer.full}>
          {nice(q.answer.label)}
        </span>
        <span class="meter min-w-0 flex-1" aria-hidden="true">
          {Array.from({ length: 10 }, (_, i) => (
            <i class={i < Math.round(q.answer.p * 10) ? 'on' : ''} style={`--row:${row};--i:${i}`}></i>
          ))}
        </span>
        <span class="w-10 text-right text-[13px] text-fg tabular-nums">{q.answer.p.toFixed(2)}</span>
      </div>
      {runnerUp ? (
        <ul class="term-fold mt-1 text-xs leading-5 text-muted" style={`--row:${row}`} role="list">
          {q.others.map((o, i) => (
            <li class={`flex items-center gap-4 ${i ? 'more' : ''}`} title={o.full}>
              <span class="w-36 shrink-0 truncate">{nice(o.label)}</span>
              <span class="meter min-w-0 flex-1 opacity-50" aria-hidden="true">
                {Array.from({ length: 10 }, (_, j) => (
                  <i class={j < Math.round(o.p * 10) ? 'on' : ''} style="animation:none"></i>
                ))}
              </span>
              <span class="w-10 text-right tabular-nums">{o.p.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </li>
  )
}

type Recording = typeof hero
const examples: Recording[] = (heroExamples as unknown[]).filter((e) => (e as { output?: unknown }).output).map((e) => heroOf(e as Parameters<typeof heroOf>[0]))

function ExamplePanel({ ex, hidden }: { ex: Recording; hidden: boolean }) {
  const n = ex.questions.length
  return (
    <div data-example hidden={hidden}>
      <div class="grid md:grid-cols-[minmax(0,1fr)_minmax(0,1.35fr)]">
        <div class="flex flex-col border-b border-line p-6 md:border-r md:border-b-0">
          <p class="text-sm text-muted">A support message comes in</p>
          <blockquote class="term-row mt-4 text-xl leading-snug font-medium text-fg md:text-[22px]" style="--row:0">
            “{ex.message}”
          </blockquote>
          <p class="mt-8 text-sm leading-relaxed text-muted md:mt-auto md:pt-8">
            Ollaya asks <span class="font-mono text-[13px] text-body">{ex.model}</span> the {n} questions of the{' '}
            <span class="font-mono text-[13px] text-body">{ex.preset}</span> preset, in one request.
          </p>
        </div>
        <ol class="divide-y divide-line" role="list">
          {ex.questions.map((q, i) => (
            <AnswerRow q={q} row={i + 1} />
          ))}
        </ol>
      </div>
      <p class="border-t border-line px-5 py-3 text-[13px] text-muted">
        Real output: {n} answers in <span class="text-fg">{ex.latencyMs} ms</span> on an {ex.device}, one forward pass. Every answer is a probability you can set a threshold on.
      </p>
    </div>
  )
}

function DecisionDemo() {
  return (
    <figure class="panel panel-strong overflow-hidden rounded-lg" style="--fold-at:3.4s" data-ui data-demo>
      <div class="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-3 text-[13px] text-muted">
        <span>{examples.length} recorded examples, the same model and preset.</span>
        <span class="flex gap-2">
          <button type="button" data-toggle aria-pressed="false" class="inline-flex h-8 items-center rounded-md border border-line px-3 text-[13px] text-body hover:border-line-strong hover:text-fg">
            Show every option
          </button>
          <button type="button" data-next class="inline-flex h-8 items-center gap-1.5 rounded-md bg-btn px-3 text-[13px] font-medium text-btn-fg hover:bg-btn-hover">
            Next example <Icon name="arrowRight" class="size-3.5" />
          </button>
        </span>
      </div>
      {examples.map((ex, i) => (
        <ExamplePanel ex={ex} hidden={i > 0} />
      ))}
    </figure>
  )
}

function Proof() {
  const stars = githubStars()
  const items = [
    [stars ?? '—', 'GitHub stars'],
    [fmt(lastWeek.pulls), 'model pulls last week'],
    [String(catalog.length), 'open decision models'],
    ['3,880', 'human-labeled questions in the bench'],
  ]
  return (
    <section class="frame" aria-label="In numbers">
      <dl class="rule grid grid-cols-2 gap-px bg-line md:grid-cols-4">
        {items.map(([v, l]) => (
          <div class="bg-canvas px-6 py-8">
            <dd class="text-4xl font-semibold tracking-[-0.03em] text-fg tabular-nums">{v}</dd>
            <dt class="mt-1 text-sm text-muted">{l}</dt>
          </div>
        ))}
      </dl>
      <div class="rule grid items-center md:grid-cols-[10rem_minmax(0,1fr)]">
        <p class="px-6 pt-7 text-[15px] text-muted md:py-9">Models by</p>
        <div class="marquee py-7 md:py-9" aria-label={PUBLISHERS.map((p) => p.name).join(', ')}>
          <div class="marquee-track" aria-hidden="true">
            {[...PUBLISHERS, ...PUBLISHERS].map((p) => (
              <span class="inline-flex shrink-0 items-center gap-3.5 text-[22px] font-medium whitespace-nowrap text-body">
                <Mark p={p} />
                {p.name}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function Title({ kicker, children, lead }: { kicker: string; children: Child; lead?: Child }) {
  return (
    <div class="rule px-6 py-14 md:px-10 md:py-20">
      <div class="max-w-3xl">
        <p class="text-[13px] tracking-[0.14em] text-muted uppercase">{kicker}</p>
        <h2 class={`mt-3 ${h2}`}>{children}</h2>
        {lead ? <p class="mt-5 max-w-2xl text-base text-body md:text-lg">{lead}</p> : null}
      </div>
    </div>
  )
}

const why = [
  {
    icon: 'adjustments' as const,
    title: 'Calibrated, not generated',
    text: 'Every option of every question gets a probability you can put a threshold on. No parsing, no hallucinated labels.',
  },
  {
    icon: 'computer' as const,
    title: 'One forward pass',
    text: `No token-by-token generation: ${result('laya:en', 'gpu.p50_ms', 0)} ms for five questions with laya:en on a GPU, and small models run well on a CPU.`,
  },
  {
    icon: 'banknotes' as const,
    title: 'Yours, for free',
    text: 'Weights from their authors, pinned by sha256. Your tickets and messages never leave the machine, and there is no per-token bill.',
  },
]

function Why() {
  return (
    <section class="frame" aria-labelledby="rd-why">
      <Title kicker="Why decision models" lead="A decision model answers typed questions about a text or a JSON: a choice, a score, a yes or no. It is the fast, reliable half of an agent.">
        <span id="rd-why">System 1 for your agents.</span>
      </Title>
      <ul class="rule grid gap-px bg-line md:grid-cols-3" role="list">
        {why.map((w) => (
          <li class="bg-canvas p-8">
            <Icon name={w.icon} class="size-6 text-fg" />
            <h3 class="mt-5 text-lg font-semibold text-fg">{w.title}</h3>
            <p class="mt-2 text-[15px] text-body">{w.text}</p>
          </li>
        ))}
      </ul>
    </section>
  )
}

type Cell = { text: string; ok?: boolean; dim?: boolean }
const rows: { label: string; ollaya: Cell; ollama: Cell; jev: Cell }[] = [
  { label: 'Best accuracy, 3,880 questions', ollaya: { text: mostAccurate.acc.toFixed(3) }, ollama: { text: vs('pplx-decider', 'ollama').acc.toFixed(3) }, jev: { text: `${JEV.acc} (typed-decisions)`, dim: true } },
  { label: 'Latency, median request', ollaya: { text: `${mostAccurate.ms} ms` }, ollama: { text: `${vs('pplx-decider', 'ollama').ms} ms` }, jev: { text: `${JEV.text}, network included` } },
  { label: 'Runs on your hardware', ollaya: { text: 'Yes', ok: true }, ollama: { text: 'Yes', ok: true }, jev: { text: 'Hosted only', ok: false } },
  ...vsFeatures.map((f) => ({ label: f.label, ollaya: { text: f.ollaya }, ollama: { text: f.ollama }, jev: { text: '—', dim: true } })),
  { label: 'Price', ollaya: { text: 'Free, Apache-2.0', ok: true }, ollama: { text: 'Free, MIT', ok: true }, jev: { text: 'Per token' } },
]

function CellView({ c, strong }: { c: Cell; strong?: boolean }) {
  return (
    <span class={`inline-flex items-start gap-2 ${strong ? 'text-fg' : c.dim ? 'text-muted' : 'text-body'}`}>
      {c.ok === undefined ? null : <Icon name={c.ok ? 'check' : 'minus'} class={`mt-1 size-4 shrink-0 ${c.ok ? 'text-pass' : 'text-faint'}`} />}
      {c.text}
    </span>
  )
}

type Dot = { name: string; server: 'ollaya' | 'ollama'; acc: number; ece: number; ms: number; label?: boolean }

/** The most accurate tag of each model on each server, as on /results, with the ECE beside it. */
function scatterDots(): Dot[] {
  const best = new Map<string, Dot>()
  for (const r of publicBench?.results ?? []) {
    if (r.errors >= r.records * 0.05) continue
    const key = `${r.server}/${shortName(r.model).split(':')[0]}`
    const b = best.get(key)
    if (!b || r.macro_accuracy > b.acc || (r.macro_accuracy === b.acc && r.median_ms < b.ms))
      best.set(key, { name: r.server === 'ollama' ? r.model : shortName(r.model), server: r.server, acc: r.macro_accuracy, ece: r.pooled_ece, ms: r.median_ms })
  }
  const dots = [...best.values()].sort((a, b) => b.acc - a.acc)
  // A label on the leaders only (Ollaya's four most accurate, Ollama's best); the rest on hover.
  let ours = 0
  let theirs = 0
  for (const d of dots) {
    if (d.server === 'ollaya' && ours < 4) d.label = ++ours > 0
    if (d.server === 'ollama' && theirs < 1) d.label = ++theirs > 0
  }
  return dots
}

const LOG_TICKS = [10, 20, 50, 100, 200, 500, 1000]
/** The model behind a tag, on either server: `kev:4b` and Ollama's `kev:4b-q8_0` are both `kev`; `clef:flash` and `clef-flash` are `clef`. */
const familyOf = (name: string) => name.split(':')[0]!.replace(/-flash$/, '')
/** Ollaya's Pareto front: the tags no other Ollaya tag beats on both accuracy and latency, fastest first. */
function frontier(dots: Dot[]): Dot[] {
  const out: Dot[] = []
  let best = -1
  for (const d of [...dots].filter((d) => d.server === 'ollaya').sort((a, b) => a.ms - b.ms)) if (d.acc > best) (best = d.acc), out.push(d)
  return out
}

function DesignScatter({ W = 760, H = 420 }: { W?: number; H?: number }) {
  const dots = scatterDots()
  const m = { l: 58, r: 20, t: 16, b: 40 }
  const xs = dots.map((d) => d.ms)
  const xTicks = LOG_TICKS.filter((t, i) => (LOG_TICKS[i + 1] ?? Infinity) > Math.min(...xs) && (LOG_TICKS[i - 1] ?? 0) < Math.max(...xs))
  const lo = Math.floor(Math.min(...dots.map((d) => d.acc)) * 10) / 10
  const hi = Math.ceil(Math.max(...dots.map((d) => d.acc)) * 10) / 10
  const yTicks = Array.from({ length: Math.round((hi - lo) * 10) + 1 }, (_, i) => lo + i / 10)
  const la = Math.log(xTicks[0]!)
  const lb = Math.log(xTicks.at(-1)!)
  const x = (v: number) => m.l + ((Math.log(Math.max(xTicks[0]!, Math.min(xTicks.at(-1)!, v))) - la) / (lb - la)) * (W - m.l - m.r)
  const y = (v: number) => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b)
  return (
    <div class="relative" data-chart>
      <div class="mb-4 flex flex-wrap items-center gap-2 text-[13px]">
        <span class="mr-2 inline-flex items-center gap-2 text-muted">
          <svg viewBox="0 0 24 12" class="h-3 w-6" aria-hidden="true">
            <line x1="0" y1="6" x2="24" y2="6" class="stroke-us" stroke-width="1.5" stroke-dasharray="4 4" />
          </svg>
          Ollaya's frontier
        </span>
        {(['ollaya', 'ollama'] as const).map((srv) => (
          <button type="button" data-series={srv} aria-pressed="true" class="inline-flex items-center gap-2 rounded-md border border-line px-2.5 py-1 text-body hover:border-line-strong aria-[pressed=false]:opacity-40">
            <svg viewBox="0 0 12 12" class="size-3" aria-hidden="true">
              {srv === 'ollaya' ? <circle cx="6" cy="6" r="5" class="fill-us" /> : <circle cx="6" cy="6" r="4" class="stroke-them" fill="none" stroke-width="2" />}
            </svg>
            {srv === 'ollaya' ? 'Ollaya' : OLLAMA}
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} class="w-full font-sans" style={`font-size:${W < 500 ? 1.6 : 1}em`} role="img" aria-label="Accuracy against median latency on Bespoke Labs' public benchmark, the most accurate tag of each model on Ollaya and on Ollama">
        {yTicks.map((t) => (
          <g>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} class="stroke-line" />
            <text x={m.l - 8} y={y(t) + 4} text-anchor="end" font-size="11" class="fill-muted">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g>
            <line x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} class="stroke-line" />
            <text x={x(t)} y={H - m.b + 16} text-anchor="middle" font-size="11" class="fill-muted">
              {t}
            </text>
          </g>
        ))}
        <text x={(m.l + W - m.r) / 2} y={H - 4} text-anchor="middle" font-size="11" class="fill-muted">
          median latency per question, ms (log scale)
        </text>
        <text transform={`translate(11 ${(m.t + H - m.b) / 2}) rotate(-90)`} text-anchor="middle" font-size="11" class="fill-muted">
          accuracy
        </text>
        <polyline
          points={frontier(dots).map((d) => `${x(d.ms).toFixed(1)},${y(d.acc).toFixed(1)}`).join(' ')}
          fill="none"
          class="stroke-us"
          stroke-width="1.5"
          stroke-dasharray="4 4"
          opacity="0.55"
          stroke-linejoin="round"
        >
          <title>Ollaya's frontier: the most accurate model at each latency</title>
        </polyline>
        {dots.map((d) => (
          <g
            class="pt"
            tabindex={0}
            data-name={d.name}
            data-family={familyOf(d.name)}
            data-x={x(d.ms).toFixed(1)}
            data-y={y(d.acc).toFixed(1)}
            data-server={d.server === 'ollaya' ? 'ollaya' : OLLAMA}
            data-acc={d.acc.toFixed(3)}
            data-ece={d.ece.toFixed(3)}
            data-ms={Math.round(d.ms)}
          >
            <title>{`${d.name} on ${d.server === 'ollaya' ? 'Ollaya' : OLLAMA}: accuracy ${d.acc.toFixed(3)}, ECE ${d.ece.toFixed(3)}, ${Math.round(d.ms)} ms`}</title>
            {d.server === 'ollaya' ? (
              <circle cx={x(d.ms)} cy={y(d.acc)} r="5" class="dot fill-us stroke-canvas" stroke-width="1.5" />
            ) : (
              <circle cx={x(d.ms)} cy={y(d.acc)} r="4.5" class="dot stroke-them fill-canvas" stroke-width="2" />
            )}
            {d.label ? (
              <text x={x(d.ms) + 9} y={y(d.acc) + 4} font-size="12" class={d.server === 'ollaya' ? 'fill-fg' : 'fill-them'}>
                {d.name}
              </text>
            ) : null}
          </g>
        ))}
      </svg>
      <div class="tip panel panel-strong rounded-md px-1 py-1 text-[14px] text-body" data-tip hidden></div>
    </div>
  )
}

function Compare() {
  return (
    <section class="frame" aria-labelledby="rd-compare">
      <Title
        kicker="Measured"
        lead={`${OLLAMA}'s nine decision models and Ollaya's, on the same ${benchGpu} and Bespoke Labs' 3,880 human-labeled questions. Ollaya runs the most accurate model and more of them, with each author's temperature. On the same weights Ollama is ${speedup()} times faster: it runs Q8_0 GGUFs where Ollaya computes the authors' BF16 weights.`}
      >
        <span id="rd-compare">Against Ollama and hosted Jev.</span>
      </Title>
      <div class="rule">
        <table class="block w-full border-collapse text-[15px] md:table md:table-fixed">
          <thead class="hidden md:table-header-group">
            <tr class="border-b border-line text-left text-[13px]">
              <th class="w-[26%] px-6 py-4 font-medium text-muted"></th>
              <th class="w-[28%] px-6 py-4 font-medium text-fg">Ollaya</th>
              <th class="px-6 py-4 font-medium text-muted">{OLLAMA}</th>
              <th class="hidden px-6 py-4 font-medium text-muted md:table-cell">TypeSafe Jev, hosted</th>
            </tr>
          </thead>
          <tbody class="block divide-y divide-line md:table-row-group">
            {rows.map((r) => (
              <tr class="block px-5 py-4 md:table-row md:p-0">
                <th scope="row" class="block pb-2 text-left align-top font-medium text-fg md:table-cell md:px-6 md:py-3.5">
                  {r.label}
                </th>
                {(
                  [
                    ['Ollaya', r.ollaya, true],
                    [OLLAMA, r.ollama, false],
                    ['TypeSafe Jev, hosted', r.jev, false],
                  ] as const
                ).map(([who, c, strong]) => (
                  <td class={`flex gap-3 py-1 align-top md:table-cell md:px-6 md:py-3.5 ${strong ? 'md:bg-subtle' : ''}`}>
                    <span class="w-28 shrink-0 text-[13px] leading-6 text-muted md:hidden">{who}</span>
                    <CellView c={c} strong={strong} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figure class="rule p-6 md:p-10">
        <figcaption class="text-sm text-body">
          <span class="font-medium text-fg">Accuracy against latency</span>, the most accurate tag of each model on each server
        </figcaption>
        <div class="mt-5 hidden md:block">
          <DesignScatter />
        </div>
        <div class="mt-5 md:hidden">
          <DesignScatter W={380} H={400} />
        </div>
        <p class="mt-6 text-sm">
          <a href={benchUrl()} class="inline-flex items-center gap-1.5 font-medium text-fg underline-offset-4 hover:underline">
            Every model on Ollaya Bench, with the raw data <Icon name="arrowRight" class="size-4" />
          </a>
        </p>
      </figure>
    </section>
  )
}

/** A code block with line numbers; each line is highlighted on its own (no token spans a line here). */
function NumberedCode({ code, lang }: { code: string; lang: string }) {
  return (
    <pre class="code overflow-x-auto py-4 pr-12 font-mono text-fg">
      <code class={`language-${lang}`}>
        {code.split('\n').map((line) => (
          <span class="ln">
            <span dangerouslySetInnerHTML={{ __html: highlight(line, lang) || ' ' }}></span>
          </span>
        ))}
      </code>
    </pre>
  )
}

function Api() {
  return (
    <section class="frame" aria-labelledby="rd-api">
      <Title kicker="Drop-in" lead="Ollaya serves TypeSafe's request and response shapes. Point the official SDK at localhost and nothing else changes.">
        <span id="rd-api">Jev-compatible, self-hosted.</span>
      </Title>
      <div class="rule grid gap-px bg-line xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        {[
          ['Request', compatRequest, 'shell'],
          ['Response', compatResponse, 'json'],
        ].map(([label, code, lang]) => (
          <div class="relative flex flex-col bg-canvas" data-copy-scope>
            <div class="border-b border-line px-5 py-3 text-xs font-medium text-muted">{label}</div>
            <NumberedCode code={code!} lang={lang!} />
            <CopyButton class="absolute top-1.5 right-2" />
          </div>
        ))}
      </div>
    </section>
  )
}

/** A model as a card: who made it, what it is, and its numbers from the bench. */
export function ModelCard({ m }: { m: (typeof catalog)[number] }) {
  const summary = (m.description.match(/^.+?[.!?](?=\s|$)/)?.[0] ?? m.description).replace(/\.$/, '')
  return (
    <a href={`/library/${m.name}`} class="group flex h-full w-full flex-col bg-canvas p-6 transition-colors hover:bg-subtle">
      <div class="flex items-baseline justify-between gap-3">
        <span class="font-mono text-[17px] text-fg group-hover:underline underline-offset-4">{m.name}</span>
        <span class="truncate text-xs text-muted">{m.publisher?.name ?? ''}</span>
      </div>
      <p class="mt-3 line-clamp-3 flex-1 text-[14px] text-body">{summary}</p>
      <dl class="mt-6 flex items-end justify-between gap-4 border-t border-line pt-4">
        <div>
          <dd class="text-2xl font-semibold tracking-[-0.02em] text-fg tabular-nums">{m.stats?.accuracy ? m.stats.accuracy.toFixed(3) : '—'}</dd>
          <dt class="text-xs text-muted">accuracy</dt>
        </div>
        <div>
          <dd class="text-2xl font-semibold tracking-[-0.02em] text-fg tabular-nums">{m.stats?.latencyMs ? `${Math.round(m.stats.latencyMs)} ms` : '—'}</dd>
          <dt class="text-xs text-muted">five questions</dt>
        </div>
        <div class="text-right text-xs text-muted">{m.sizes.join(' · ')}</div>
      </dl>
    </a>
  )
}

function Models() {
  const top = [...catalog].sort((a, b) => a.rank - b.rank).slice(0, 6)
  return (
    <section class="frame" aria-labelledby="rd-models">
      <Title kicker="Open models" lead={`${catalog.length} families from their authors, pulled by name. Every number here is from Ollaya Bench on an ${benchGpu}.`}>
        <span id="rd-models">Open weights, ready to pull.</span>
      </Title>
      <ul class="rule grid gap-px bg-line sm:grid-cols-2 lg:grid-cols-3" role="list">
        {top.map((m) => (
          <li class="flex">
            <ModelCard m={m} />
          </li>
        ))}
      </ul>
      <p class="rule px-6 py-5 text-sm">
        <a href="/search" class="inline-flex items-center gap-1.5 font-medium text-fg underline-offset-4 hover:underline">
          Browse all {catalog.length} models <Icon name="arrowRight" class="size-4" />
        </a>
      </p>
    </section>
  )
}

function Closer({ origin }: { origin: string }) {
  return (
    <section class="relative overflow-hidden" aria-labelledby="rd-closer">
      <div class="field" aria-hidden="true">
        <i class="a" style="top:auto;bottom:-30vw"></i>
        <i class="b" style="top:auto;bottom:-20vw"></i>
      </div>
      <div class="frame rule relative px-5 py-28 text-center">
        <LogoMark class="mx-auto size-14 text-fg" />
        <h2 id="rd-closer" class={`mt-6 ${h2} md:text-6xl`}>
          Up and running in a minute.
        </h2>
        <p class="mx-auto mt-5 max-w-md text-body">macOS, Windows, Linux and Docker. CPU, NVIDIA and Apple GPUs.</p>
        <div class="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Install origin={origin} />
          <a href="/download" class="inline-flex h-11 items-center gap-2 rounded-md bg-btn px-5 text-sm font-medium text-btn-fg hover:bg-btn-hover">
            <Icon name="download" class="size-4" />
            Download
          </a>
        </div>
      </div>
    </section>
  )
}

type Support = { ok: boolean; text?: string; note?: string }

// What each release ships (see /download and the release assets). GPU means a provider the runner
// registers: CUDA on NVIDIA, and MLX on Apple silicon for the layouts that pass parity on Metal.
const platforms: { name: string; detail: string; app: Support; cli: Support; gpu: Support }[] = [
  {
    name: 'macOS',
    detail: 'Apple silicon, macOS 14+',
    app: { ok: true, text: 'Menu bar app', note: '.dmg' },
    cli: { ok: true, text: 'Install script' },
    gpu: { ok: true, text: 'Apple GPU', note: 'Laya and NLI on MLX' },
  },
  {
    name: 'Windows',
    detail: '10 and 11, x64',
    app: { ok: true, text: 'Desktop app', note: '.exe or .msi' },
    cli: { ok: true, text: 'PowerShell script' },
    gpu: { ok: true, text: 'NVIDIA, CUDA 12/13', note: 'Command line' },
  },
  {
    name: 'Linux',
    detail: 'x86-64',
    app: { ok: true, text: 'Desktop app', note: 'AppImage, .deb, .rpm' },
    cli: { ok: true, text: 'Install script', note: 'systemd service' },
    gpu: { ok: true, text: 'NVIDIA, CUDA 12/13' },
  },
  {
    name: 'Linux',
    detail: 'ARM64',
    app: { ok: false },
    cli: { ok: true, text: 'Install script', note: 'systemd service' },
    gpu: { ok: false, text: 'CPU only' },
  },
  {
    name: 'WSL 2',
    detail: 'Linux on Windows',
    app: { ok: false },
    cli: { ok: true, text: 'Install script', note: 'Same as Linux' },
    gpu: { ok: true, text: 'NVIDIA, CUDA 12/13' },
  },
  {
    name: 'Docker',
    detail: 'amd64 and arm64',
    app: { ok: false },
    cli: { ok: true, text: 'Image on GHCR' },
    gpu: { ok: true, text: 'NVIDIA, CUDA 12/13', note: ':cuda and :cuda12, amd64' },
  },
]

const platformColumns = ['Desktop app', 'Command line', 'GPU'] as const

/** A table from md up; below that each row stacks into labelled lines. */
function SupportCell({ label, s }: { label: string; s: Support }) {
  return (
    <td class="flex gap-4 pt-2.5 md:table-cell md:px-4 md:py-5 xl:px-6 md:align-top">
      <span class="w-28 shrink-0 text-[13px] leading-6 text-muted md:hidden">{label}</span>
      <span class="flex gap-2.5">
        <Icon name={s.ok ? 'check' : 'minus'} class={`mt-1 size-4 shrink-0 ${s.ok ? 'text-fg' : 'text-faint'}`} />
        <span class="text-[15px] leading-6">
          {s.text ? <span class={s.ok ? 'text-body' : 'text-muted'}>{s.text}</span> : <span class="sr-only">Not available</span>}
          {s.note && <span class="block text-[13px] leading-5 text-muted">{s.note}</span>}
        </span>
      </span>
    </td>
  )
}

function PlatformsSection() {
  return (
    <section class="frame" aria-labelledby="platforms">
      <Title kicker="Platforms" lead="A desktop app and a command line for macOS, Windows and Linux, and a Docker image for servers. Every model runs on the CPU; an NVIDIA GPU on Linux, Windows, WSL 2 or Docker takes a request down to milliseconds.">
        <span id="platforms">Runs where you work.</span>
      </Title>
      <div class="rule">
        <table class="block w-full border-collapse md:table md:table-fixed">
          <thead class="hidden border-b border-line bg-subtle md:table-header-group">
            <tr>
              <th scope="col" class="w-[20%] px-6 py-3 text-left md:px-4 xl:px-6 text-[13px] font-medium text-muted">
                Platform
              </th>
              {platformColumns.map((c) => (
                <th scope="col" class="px-6 py-3 text-left text-[13px] md:px-4 xl:px-6 font-medium text-muted">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody class="block divide-y divide-line md:table-row-group">
            {platforms.map((p) => (
              <tr class="block px-5 py-5 md:table-row md:p-0">
                <th scope="row" class="block pb-1 text-left font-normal md:table-cell md:px-4 md:py-5 xl:px-6 md:align-top">
                  <span class="block text-[15px] leading-6 font-medium text-fg">{p.name}</span>
                  <span class="block text-[13px] leading-5 text-muted">{p.detail}</span>
                </th>
                <SupportCell label={platformColumns[0]} s={p.app} />
                <SupportCell label={platformColumns[1]} s={p.cli} />
                <SupportCell label={platformColumns[2]} s={p.gpu} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div class="rule flex flex-col gap-4 px-6 py-5 sm:flex-row sm:items-start sm:justify-between">
        <a href="/download" class="inline-flex items-center gap-1.5 text-sm font-medium text-fg underline-offset-4 hover:underline">
          Install for your platform <Icon name="arrowRight" class="size-4" />
        </a>
        <p class="max-w-md text-[13px] text-muted">
          NVIDIA GPUs need driver R525 or newer; the install scripts fetch the CUDA libraries only when they find one.
          On a Mac, laya and nli run on the Apple GPU through MLX; other models, AMD and Intel GPUs, and the Windows and Linux desktop apps without the command line installed use the CPU.
        </p>
      </div>
    </section>
  )
}

export function HomePage({ origin }: { origin: string }) {
  return (
    <>
      <Hero origin={origin} />
      <Proof />
      <Why />
      <Compare />
      <Api />
      <Models />
      <PlatformsSection />
      <Closer origin={origin} />
    </>
  )
}
