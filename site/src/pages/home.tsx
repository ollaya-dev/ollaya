import type { Child } from 'hono/jsx'
import { Icon, type IconName } from '../components/Icon'
import { LogoMark } from '../components/Logo'
import { Legend } from '../components/charts'
import { btnPrimary, Code, CodeBlock, textLink } from '../components/ui'
import { catalog, comingNext, featuredTags, fullName } from '../data/catalog'
import { GITHUB_URL, LOCAL_API } from '../site'

const sections = [
  { id: 'vs-ollama', label: 'Ollaya and Ollama' },
  { id: 'fast', label: 'Fast and accurate' },
  { id: 'compatible', label: 'Drop-in compatible' },
  { id: 'models', label: 'Open models' },
  { id: 'private', label: 'Your data stays yours' },
  { id: 'platforms', label: 'Platforms' },
]

export function HomePage() {
  return (
    <>
      <Hero />
      <div class="mx-auto mt-24 max-w-6xl px-4 md:mt-36 md:px-6 lg:grid lg:grid-cols-[11rem_minmax(0,1fr)] lg:gap-16">
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
        <div class="space-y-24 md:space-y-36">
          <VersusOllama />
          <Fast />
          <Compatible />
          <OpenModels />
          <Private />
          <Platforms />
        </div>
      </div>
      <Closer />
    </>
  )
}

// ---------------------------------------------------------------------------------------------

function Hero() {
  return (
    <section class="mx-auto max-w-6xl px-4 pt-10 md:px-6 md:pt-20" aria-labelledby="hero-title">
      <div class="grid items-center gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] lg:gap-16">
        <div>
          <h1
            id="hero-title"
            class="text-4xl leading-[1.05] font-medium tracking-tight text-fg md:text-5xl lg:text-[3.5rem]"
          >
            Run decision models locally.
          </h1>
          <p class="mt-5 max-w-xl text-lg text-body">
            Ask typed questions about any text or JSON and get calibrated answers in milliseconds. Private, open
            source, on your own hardware.
          </p>
          <div class="mt-8 flex flex-wrap items-center gap-x-3 gap-y-3">
            <a href="/download" class={btnPrimary}>
              <Icon name="download" class="size-4" />
              Download
            </a>
            <a
              href="/search"
              class="inline-flex items-center gap-1.5 rounded-full px-3 py-2.5 text-sm font-medium text-fg underline-offset-4 hover:underline"
            >
              Browse models <Icon name="arrowRight" class="size-4" />
            </a>
          </div>
          <p class="mt-6 text-xs text-muted">
            An independent open-source project, not affiliated with Ollama or TypeSafe.
          </p>
        </div>
        <TerminalMock />
      </div>
    </section>
  )
}

// Real output of the command shown (the default triage preset), run on an RTX 4090 through the CLI
// with `--verbose` timings: winnow:e4b answered its five questions in 87 ms (median of ten warm
// runs). The first line holds the prompt; the others hang under it.
// The command, then the state: a quoted message that continues on the next lines. Short lines, so
// nothing wraps on a phone.
const MOCK_COMMAND = 'ollaya run winnow:e4b \\'
const MOCK_MESSAGE = [
  '"Third time this year you\'ve',
  'double-charged me. Refund it',
  'today or I\'m cancelling and',
  'moving to a competitor."',
]
const mockRows = [
  { q: 'intent', a: 'refund', p: 0.91 },
  { q: 'is_urgent', a: 'yes', p: 0.92 },
  { q: 'frustration', a: '2.89 / 3', note: 'very angry', p: 0.86 },
  { q: 'refund_requested', a: 'yes', p: 0.99 },
  { q: 'churn_risk', a: 'yes', p: 0.99 },
]

/** One command line: the prompt in its own column, so continuation lines hang under the command. */
function PromptLine({ children }: { children?: Child }) {
  return (
    <div class="flex gap-[1ch]">
      <span class="text-muted select-none" aria-hidden="true">
        $
      </span>
      <div class="min-w-0">{children}</div>
    </div>
  )
}

function TerminalMock() {
  return (
    // On wide screens the caption is taken out of the flow, so the hero text centres on the window.
    <figure class="relative min-w-0">
      <div class="term-glow overflow-hidden rounded-xl border border-line bg-term">
        <div class="relative flex h-10 items-center border-b border-line px-4 sm:px-5" aria-hidden="true">
          <span class="flex gap-2">
            <span class="size-3 rounded-full bg-[#ff5f57]"></span>
            <span class="size-3 rounded-full bg-[#febc2e]"></span>
            <span class="size-3 rounded-full bg-[#28c840]"></span>
          </span>
          <span class="absolute inset-x-0 text-center text-xs text-muted">ollaya · zsh</span>
        </div>
        <div class="p-4 font-mono text-[12.5px] leading-6 text-fg sm:p-5 sm:text-[13px]">
          <PromptLine>
            <pre class="whitespace-pre-wrap">
              <Code code={MOCK_COMMAND} lang="shell" />
            </pre>
            {MOCK_MESSAGE.map((line, i) => (
              <pre class={`whitespace-pre-wrap ${i ? 'pl-[3ch]' : 'pl-[2ch]'}`}>
                <span class="tok-string">{line}</span>
              </pre>
            ))}
          </PromptLine>
          <table class="mt-4 w-full border-collapse text-left">
            <caption class="sr-only">Answers returned by the model</caption>
            <thead class="sr-only">
              <tr>
                <th scope="col">Question</th>
                <th scope="col">Answer</th>
                <th scope="col">Probability</th>
              </tr>
            </thead>
            <tbody>
              {mockRows.map((r, i) => (
                <tr class="term-row" style={`--row:${i}`}>
                  <td class="w-[18ch] py-0.5 pr-4 align-middle whitespace-nowrap text-muted">{r.q}</td>
                  <td class="py-0.5 pr-4 align-middle font-medium whitespace-nowrap">
                    {r.a}
                    {r.note ? <span class="hidden font-normal text-muted sm:inline">{`  ${r.note}`}</span> : null}
                  </td>
                  <td class="w-0 py-0.5 align-middle">
                    <span class="flex items-center justify-end gap-3">
                      <span
                        class="hidden h-1.5 w-16 overflow-hidden rounded-full bg-fill-strong min-[420px]:block sm:w-20"
                        aria-hidden="true"
                      >
                        <span class="term-bar block h-full rounded-full" style={`width:${Math.round(r.p * 100)}%`}></span>
                      </span>
                      <span class="tabular-nums">{r.p.toFixed(2)}</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div class="term-row mt-4" style={`--row:${mockRows.length}`}>
            <PromptLine>
              <span class="term-cursor inline-block h-[1.15em] w-[1ch] translate-y-[0.2em] bg-fg/80" aria-hidden="true"></span>
            </PromptLine>
          </div>
        </div>
      </div>
      <figcaption class="mt-3 text-center text-xs text-muted lg:absolute lg:inset-x-0 lg:top-full">
        Real output: <span class="font-mono">winnow:e4b</span> answered five questions in 87 ms on an RTX 4090.
      </figcaption>
    </figure>
  )
}

// ---------------------------------------------------------------------------------------------

function Section({
  id,
  title,
  lead,
  children,
  body,
}: {
  id: string
  title: string
  lead: string
  body: Child
  children: Child
}) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} class="scroll-mt-24" data-section>
      <h2 id={`${id}-title`} class="text-base font-semibold text-fg">
        {title}
      </h2>
      <p class="mt-2 max-w-2xl text-2xl leading-tight font-medium tracking-tight text-fg md:text-3xl">{lead}</p>
      <p class="mt-4 max-w-2xl text-base text-body md:text-lg">{body}</p>
      <div class="mt-10">{children}</div>
    </section>
  )
}

// Accuracy: the typed-decisions test split (400 states, 2,000 questions, argmax against the
// majority label), as published on each model's page; Jev's is from Winnow's benchmark report on
// the same 2,000 questions. Latency: median of 15 warm five-question requests (the triage preset on
// the hero's message) through the HTTP API on an RTX 4090, each model in its shipped precision;
// Jev's is the hosted API's median request in third-party benchmarks, network included.
// laya:typed-decisions (0.766) and jeb (0.79-0.80) are left out: both were trained on this dataset's train split. clm's latency is a new
// message whose five questions are already cached (a repeated request takes under a millisecond).
const scoreboard: { tag: string; acc: number; ms: number; pick?: boolean; note?: string }[] = [
  { tag: 'winnow:e4b', acc: 0.722, ms: 89, pick: true },
  { tag: 'kev:9b', acc: 0.722, ms: 498 },
  { tag: 'winnow:12b', acc: 0.702, ms: 131 },
  { tag: 'cygnet:12b', acc: 0.683, ms: 202 },
  { tag: 'decider:4b', acc: 0.68, ms: 520 },
  { tag: 'jeeves:9b', acc: 0.68, ms: 838 },
  { tag: 'kev:4b', acc: 0.669, ms: 354 },
  { tag: 'nimble:9b', acc: 0.665, ms: 2297 },
  { tag: 'jevk5:4b', acc: 0.625, ms: 105 },
  { tag: 'decider:2b', acc: 0.591, ms: 190 },
  { tag: 'nli', acc: 0.548, ms: 20 },
  { tag: 'decider:0.8b', acc: 0.506, ms: 155 },
  { tag: 'gliclass', acc: 0.477, ms: 15 },
  { tag: 'kev:0.8b', acc: 0.46, ms: 128 },
  { tag: 'von', acc: 0.447, ms: 23 },
  { tag: 'laya:en', acc: 0.361, ms: 10 },
  { tag: 'clm:8b', acc: 0.357, ms: 149, note: 'questions cached' },
]
const JEV = { acc: 0.738, text: '236–276 ms' }
const ACC_SCALE = 0.8
const ACC_AXIS = [0, 0.2, 0.4, 0.6, 0.8]
const accPct = (a: number) => `${((a / ACC_SCALE) * 100).toFixed(2)}%`
const msText = (ms: number) => `${ms} ms`

function Stat({ value, unit, label, detail, muted }: { value: string; unit: string; label: string; detail: string; muted?: boolean }) {
  return (
    <div class="px-6 py-7 sm:px-8">
      <p
        class={`text-5xl font-semibold tracking-tight whitespace-nowrap tabular-nums sm:text-[2.75rem] lg:text-6xl ${muted ? 'text-muted' : 'text-fg'}`}
      >
        {value}
        <span class="ml-1.5 text-2xl font-medium tracking-normal lg:text-3xl">{unit}</span>
      </p>
      <p class="mt-3 text-sm font-medium text-fg">{label}</p>
      <p class="mt-0.5 text-sm text-muted">{detail}</p>
    </div>
  )
}

/** One scoreboard row: the tag, an accuracy bar with its value, and the latency. */
function ScoreRow({ label, note, acc, latency, pick, muted }: { label: string; note?: string; acc: number; latency: string; pick?: boolean; muted?: boolean }) {
  return (
    <li class="contents">
      <span class="flex h-9 flex-col justify-center leading-tight">
        <span class={`font-mono text-xs sm:text-[13px] ${muted ? 'font-sans text-body' : 'text-fg'} ${pick ? 'font-semibold' : ''}`}>{label}</span>
        {note ? <span class="text-xs text-muted">{note}</span> : null}
      </span>
      <span class="relative flex h-9 items-center">
        <span class="min-w-0 flex-1">
          <span class={`block h-2.5 min-w-1 rounded-full ${muted ? 'bg-them' : 'bg-us'}`} style={`width:${accPct(acc)}`}></span>
        </span>
        <span class={`ml-2.5 w-11 text-[13px] whitespace-nowrap tabular-nums ${muted ? 'text-body' : 'font-medium text-fg'}`}>{acc.toFixed(3)}</span>
      </span>
      <span class={`flex h-9 items-center justify-end text-[13px] whitespace-nowrap tabular-nums ${muted ? 'text-body' : 'text-fg'} ${pick ? 'font-semibold' : ''}`}>
        {latency}
      </span>
    </li>
  )
}

function Fast() {
  return (
    <Section
      id="fast"
      title="Fast and accurate"
      lead="Close to Jev's accuracy, in under 100 ms."
      body="A decision model answers in a single forward pass, with no token-by-token generation. On an RTX 4090, winnow:e4b answers a five-question request in 89 ms end to end, and scores 0.722 on typed decisions against 0.738 for TypeSafe's hosted Jev. Smaller models such as laya answer in about 10 ms, and run well on a CPU."
    >
      <div class="grid overflow-hidden rounded-2xl border border-line sm:grid-cols-2">
        <Stat value="89" unit="ms" label="winnow:e4b on Ollaya" detail="RTX 4090, five questions · 0.722 accuracy" />
        <div class="border-t border-line sm:border-t-0 sm:border-l">
          <Stat value="236–276" unit="ms" label="TypeSafe Jev" detail="Hosted API, median request · 0.738 accuracy" muted />
        </div>
      </div>

      <figure class="mt-14">
        <figcaption class="text-sm font-medium text-fg">
          Accuracy and speed of every model{' '}
          <span class="font-normal text-muted">· typed-decisions accuracy, higher is better; latency, lower is better</span>
        </figcaption>
        <div class="mt-4">
          <Legend
            items={[
              { label: 'Ollaya', tone: 'us' },
              { label: "TypeSafe's hosted Jev", tone: 'them' },
            ]}
          />
        </div>
        <div class="mt-6 grid grid-cols-[6.5rem_minmax(0,1fr)_5.25rem] gap-x-3 sm:grid-cols-[9.5rem_minmax(0,1fr)_5.5rem] sm:gap-x-4">
          <span class="text-xs text-muted">Model</span>
          <span class="text-xs text-muted">Accuracy</span>
          <span class="text-right text-xs text-muted">Latency</span>
          <ul class="contents" role="list">
            <ScoreRow label="TypeSafe Jev" note="hosted API" acc={JEV.acc} latency={JEV.text} muted />
            {scoreboard.map((r) => (
              <ScoreRow label={r.tag} note={r.note} acc={r.acc} latency={msText(r.ms)} pick={r.pick} />
            ))}
          </ul>
          <span></span>
          {/* The axis spans the bars' track: the column less the value beside each bar (ml-2.5 + w-11). */}
          <span class="relative mt-2 mr-[3.375rem] h-5 border-t border-line text-[11px] text-muted tabular-nums" aria-hidden="true">
            {ACC_AXIS.map((t) => (
              <span
                class={`absolute top-1.5 whitespace-nowrap ${t === 0 ? '' : t === ACC_SCALE ? '-translate-x-full' : '-translate-x-1/2'}`}
                style={`left:${accPct(t)}`}
              >
                {t === 0 ? '0' : t.toFixed(1)}
              </span>
            ))}
          </span>
          <span></span>
        </div>
        <p class="mt-8 max-w-2xl text-[13px] leading-relaxed text-muted">
          Accuracy: the typed-decisions test split (400 states, 2,000 questions) against the majority label; Jev's from Winnow's
          report on the same questions. <span class="font-mono">laya:typed-decisions</span> and <span class="font-mono">jeb</span>{' '}
          were trained on this dataset, so they are left out. Latency: the median five-question request through the HTTP API on an
          RTX 4090 (<span class="font-mono">clm</span> with its questions cached); Jev's is the hosted API in third-party benchmarks (
          <a href="https://github.com/AbdelStark/jev-benchmarks" class={textLink}>
            AbdelStark/jev-benchmarks
          </a>
          ,{' '}
          <a href="https://github.com/nibzard/decision-model-benchmark" class={textLink}>
            nibzard/decision-model-benchmark
          </a>
          ), network included, so compare orders of magnitude.
        </p>
      </figure>
    </Section>
  )
}

// Bespoke Labs' public decision benchmark (github.com/bespokelabsai/nimble, docs/PUBLIC_BENCHMARKS.md):
// 13 subsets of 11 human-labeled datasets, 3,880 questions, rebuilt byte for byte from Bespoke's manifests
// and scored with Bespoke's own runner (convert/ollaya_convert/bench_public.py). Both servers ran on the same
// RTX 4090, one request at a time through /v1/systemone. Accuracy: macro average over the 13 subsets. ECE:
// ten-bin expected calibration error of the top probability, averaged over the subsets (lower is better).
// Latency: median request, HTTP included. Numbers: results/runs/2026-09-30-public-benchmark-rtx5090-cuda.json.
type VsRun = { tag: string; server: 'ollaya' | 'ollama'; acc: number; ece: number; ms: number }
const vsRuns: VsRun[] = [
  { tag: 'winnow:12b', server: 'ollaya', acc: 0.773, ece: 0.141, ms: 60 },
  { tag: 'decider:4b', server: 'ollaya', acc: 0.756, ece: 0.043, ms: 188 },
  { tag: 'kev:9b', server: 'ollaya', acc: 0.753, ece: 0.058, ms: 229 },
  { tag: 'nimble', server: 'ollama', acc: 0.749, ece: 0.122, ms: 210 },
  { tag: 'nimble:9b', server: 'ollaya', acc: 0.748, ece: 0.022, ms: 310 },
  { tag: 'tev1:4b', server: 'ollama', acc: 0.747, ece: 0.075, ms: 406 },
  { tag: 'winnow:e4b', server: 'ollaya', acc: 0.734, ece: 0.071, ms: 43 },
  { tag: 'decider:2b', server: 'ollaya', acc: 0.703, ece: 0.087, ms: 98 },
  { tag: 'tev1:0.8b', server: 'ollama', acc: 0.639, ece: 0.129, ms: 61 },
  { tag: 'laya:multilingual', server: 'ollaya', acc: 0.579, ece: 0.155, ms: 14 },
]
// The accuracy axis of the comparison: a dot on a focused range, not a bar from zero, so the
// differences between 0.58 and 0.77 are visible; the ticks print the range.
const VS_MIN = 0.55
const VS_MAX = 0.8
const VS_TICKS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
const vsPos = (a: number) => `${(((Math.min(Math.max(a, VS_MIN), VS_MAX) - VS_MIN) / (VS_MAX - VS_MIN)) * 100).toFixed(2)}%`

const vsFeatures: { label: string; ollaya: string; ollama: string }[] = [
  { label: 'Decision models', ollaya: '15 families: encoders (laya, nli, gliclass, von) and decoders (winnow, kev, decider, nimble, jeb, jeeves, cygnet and more)', ollama: 'Nimble and Tev1, decoders only' },
  { label: 'Small encoders (milliseconds, CPU-friendly)', ollaya: 'laya, nli, gliclass, von', ollama: 'None' },
  { label: 'Probabilities', ollaya: "Calibrated with each author's fitted temperature, refittable in a Modelfile", ollama: 'Raw softmax; documented as uncalibrated' },
  { label: 'Options per question', ollaya: 'Up to 255, as TypeSafe', ollama: 'Up to 26' },
  { label: 'Questions per request', ollaya: 'Up to 256, as TypeSafe', ollama: 'Up to 64' },
  { label: 'TypeSafe endpoints', ollaya: '/v1/systemone, /v1/decisions, /v1/models', ollama: '/v1/systemone' },
  { label: 'Language routing', ollaya: 'laya picks English or multilingual per request', ollama: 'None' },
  { label: 'Weights', ollaya: "The author's files, pinned by commit and sha256", ollama: 'Converted to GGUF and re-hosted' },
]

/**
 * One comparison row. Each row is its own grid with the same columns, so the accuracy track can
 * drop to a line of its own on a phone (model and the three numbers on top, the full-width track
 * under them) and sit between the model and the numbers from sm up.
 */
function VsBar({ run }: { run: VsRun }) {
  const ollama = run.server === 'ollama'
  const num = `flex min-h-9 items-center justify-end text-[13px] whitespace-nowrap tabular-nums sm:order-none ${ollama ? 'text-body' : 'text-fg'}`
  return (
    <li class={VS_GRID}>
      <span class="order-1 flex min-h-9 flex-col justify-center leading-tight sm:order-none">
        <span class="font-mono text-xs text-fg sm:text-[13px]">{run.tag}</span>
        <span class="text-xs text-muted">{ollama ? 'on Ollama 0.35' : 'on Ollaya'}</span>
      </span>
      <span class="relative order-5 col-span-4 flex h-6 items-center sm:order-none sm:col-span-1 sm:h-9" aria-hidden="true">
        <span class="relative h-px w-full bg-line">
          <span
            class={`absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full ${ollama ? 'border-[2.5px] border-them bg-canvas' : 'bg-us'}`}
            style={`left:${vsPos(run.acc)}`}
          ></span>
        </span>
      </span>
      <span class={`order-2 ${num} ${ollama ? '' : 'font-medium'}`}>{run.acc.toFixed(3)}</span>
      <span class={`order-3 ${num}`}>{run.ece.toFixed(3)}</span>
      <span class={`order-4 ${num}`}>{run.ms} ms</span>
    </li>
  )
}

const VS_GRID =
  'grid grid-cols-[minmax(0,1fr)_3.25rem_3.25rem_4.25rem] gap-x-3 sm:grid-cols-[9.5rem_minmax(0,1fr)_3.75rem_3.75rem_4.75rem] sm:gap-x-4'

function VersusOllama() {
  return (
    <Section
      id="vs-ollama"
      title="Ollaya and Ollama"
      lead="Inspired by Ollama, measured side by side."
      body="Ollaya borrows Ollama's design: one binary, pull and run, a local API. Ollama 0.35 now serves two decision models too, Nimble and Tev1, so we ran both on the same RTX 5090 over Bespoke Labs' public benchmark: 3,880 human-labeled questions from 13 datasets, scored with Bespoke's own code. Ollaya leads on accuracy, calibration and the models it runs; on the same Nimble weights, Ollama is faster."
    >
      <div class="grid overflow-hidden rounded-2xl border border-line sm:grid-cols-3">
        <Stat value="0.773" unit="" label="Most accurate: winnow:12b on Ollaya" detail="Ollama's best: 0.749 (Nimble)" />
        <div class="border-t border-line sm:border-t-0 sm:border-l">
          <Stat value="60" unit="ms" label="winnow:12b per question" detail="Nimble on Ollama: 210 ms" />
        </div>
        <div class="border-t border-line sm:border-t-0 sm:border-l">
          <Stat value="5.5×" unit="" label="Lower calibration error, same Nimble" detail="ECE 0.022 on Ollaya, 0.122 on Ollama" />
        </div>
      </div>

      <figure class="mt-14">
        <figcaption class="text-sm font-medium text-fg">
          Same GPU, same 3,880 human-labeled questions{' '}
          <span class="font-normal text-muted">· accuracy, higher is better; calibration error and latency, lower is better</span>
        </figcaption>
        <div class="mt-4">
          <Legend
            items={[
              { label: 'Ollaya 0.8.0', tone: 'us' },
              { label: 'Ollama 0.35.0', tone: 'them', mark: 'ring' },
            ]}
          />
        </div>
        <div class="mt-6">
          <div class={VS_GRID} aria-hidden="true">
            <span class="text-xs text-muted">Model</span>
            <span class="hidden text-xs text-muted sm:block">Accuracy</span>
            <span class="text-right text-xs text-muted">
              <span class="sm:hidden">Acc.</span>
            </span>
            <span class="text-right text-xs text-muted">ECE</span>
            <span class="text-right text-xs text-muted">Latency</span>
          </div>
          <ul class="mt-1 space-y-2 sm:space-y-0" role="list">
            {vsRuns.map((r) => (
              <VsBar run={r} />
            ))}
          </ul>
          <div class={VS_GRID} aria-hidden="true">
            <span class="hidden sm:block"></span>
            <span class="relative col-span-4 mt-2 h-5 border-t border-line text-[11px] text-muted tabular-nums sm:col-span-1">
              {VS_TICKS.map((t, i) => (
                <span
                  class={`absolute top-1.5 ${i === 0 ? '' : i === VS_TICKS.length - 1 ? '-translate-x-full' : '-translate-x-1/2'}`}
                  style={`left:${vsPos(t)}`}
                >
                  {t.toFixed(2)}
                </span>
              ))}
            </span>
          </div>
        </div>
      </figure>

      <div class="mt-14 overflow-hidden rounded-2xl border border-line">
        <table class="block w-full border-collapse md:table md:table-fixed">
          <thead class="hidden border-b border-line bg-subtle md:table-header-group">
            <tr>
              <th scope="col" class="w-[30%] px-6 py-3 text-left text-[13px] font-medium text-muted"></th>
              <th scope="col" class="px-6 py-3 text-left text-[13px] font-medium text-fg">Ollaya</th>
              <th scope="col" class="px-6 py-3 text-left text-[13px] font-medium text-muted">Ollama 0.35</th>
            </tr>
          </thead>
          <tbody class="block divide-y divide-line md:table-row-group">
            {vsFeatures.map((f) => (
              <tr class="block px-5 py-4 md:table-row md:p-0">
                <th scope="row" class="block pb-1 text-left text-[15px] font-medium text-fg md:table-cell md:px-6 md:py-4 md:align-top">
                  {f.label}
                </th>
                <td class="flex gap-3 pt-1.5 md:table-cell md:px-6 md:py-4 md:align-top">
                  <span class="w-24 shrink-0 text-[13px] leading-6 text-muted md:hidden">Ollaya</span>
                  <span class="text-[15px] leading-6 text-body">{f.ollaya}</span>
                </td>
                <td class="flex gap-3 pt-1.5 md:table-cell md:px-6 md:py-4 md:align-top">
                  <span class="w-24 shrink-0 text-[13px] leading-6 text-muted md:hidden">Ollama 0.35</span>
                  <span class="text-[15px] leading-6 text-muted">{f.ollama}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p class="mt-8 max-w-2xl text-[13px] leading-relaxed text-muted">
        <a href="https://github.com/bespokelabsai/nimble/blob/main/docs/PUBLIC_BENCHMARKS.md" class={textLink}>Bespoke Labs' public benchmark</a>,
        one request at a time with Ollaya 0.8.0 and Ollama 0.35.0. Accuracy is the mean over the 13 datasets; ECE is the
        calibration error of the top probability; latency is the median request, HTTP included. Ollama runs Nimble as Q8_0 on
        llama.cpp, Ollaya in fp32 with the author's temperature.
      </p>
      <p class="mt-4">
        <a href="/results" class="inline-flex items-center gap-1.5 text-sm font-medium text-fg underline-offset-4 hover:underline">
          All results, every machine, with the raw data <Icon name="arrowRight" class="size-4" />
        </a>
      </p>
    </Section>
  )
}

const compatRequest = `# Point the TypeSafe SDK at Ollaya
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

const compatResponse = `{
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

function Compatible() {
  return (
    <Section
      id="compatible"
      title="Drop-in compatible"
      lead="Speaks TypeSafe's API."
      body={
        <>
          Ollaya serves <code class="font-mono text-[0.9em]">/v1/systemone</code> and{' '}
          <code class="font-mono text-[0.9em]">/v1/models</code> with TypeSafe's request and response shapes. The
          official TypeSafe Python SDK 0.7.1 works unchanged against a local server.
        </>
      }
    >
      <div class="grid grid-cols-[minmax(0,1fr)] gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div class="flex flex-col">
          <p class="mb-2 text-[13px] font-medium text-muted">Request</p>
          <CodeBlock code={compatRequest} class="flex-1" />
        </div>
        <div class="flex flex-col">
          <p class="mb-2 text-[13px] font-medium text-muted">Response</p>
          <CodeBlock code={compatResponse} lang="json" class="flex-1" />
        </div>
      </div>
      <p class="mt-6 text-sm">
        <a href="/docs/typesafe-compatibility" class="inline-flex items-center gap-1.5 font-medium text-fg underline-offset-4 hover:underline">
          TypeSafe compatibility guide <Icon name="arrowRight" class="size-4" />
        </a>
      </p>
    </Section>
  )
}

/** One row per model, or one row per featured tag while the registry holds a single family. */
function modelRows(): { href: string; name: string; summary: string; meta: string }[] {
  if (catalog.length === 1) {
    const m = catalog[0]!
    return featuredTags(m).map((t) => ({
      href: `/library/${fullName(m, t)}`,
      name: fullName(m, t),
      summary: t.summary,
      meta: t.kind === 'router' ? 'router' : [t.params, t.context && `${t.context} ctx`].filter(Boolean).join(' · '),
    }))
  }
  return catalog.slice(0, 8).map((m) => ({
    href: `/library/${m.name}`,
    name: m.name,
    summary: m.description,
    meta: m.sizes.join(' · '),
  }))
}

function OpenModels() {
  const rows = modelRows()
  const laya = catalog.some((m) => m.name === 'laya')
  return (
    <Section
      id="models"
      title="Open models"
      lead="Open weights, ready to pull."
      body={
        laya
          ? 'Pick by what you need: winnow:e4b balances accuracy and speed best, laya is the fastest and runs well on a CPU, kev and decider scale up to 9B and 4B, von reads up to 8,192 tokens, and qwen3guard screens text for safety. The models page shows each one’s accuracy and speed.'
          : 'Open decision models from their authors, pulled by name.'
      }
    >
      <ul class="divide-y divide-line border-y border-line" role="list">
        {rows.map((r) => (
          <li>
            <a href={r.href} class="group flex flex-col gap-1 py-5 sm:flex-row sm:items-baseline sm:gap-6">
              <span class="shrink-0 font-mono text-[15px] text-fg underline-offset-4 group-hover:underline sm:w-52">
                {r.name}
              </span>
              <span class="flex-1 text-body">{r.summary}</span>
              <span class="text-[13px] whitespace-nowrap text-muted">{r.meta}</span>
            </a>
          </li>
        ))}
      </ul>
      <div class="mt-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <a href="/search" class="inline-flex items-center gap-1.5 text-sm font-medium text-fg underline-offset-4 hover:underline">
          Browse all models <Icon name="arrowRight" class="size-4" />
        </a>
        {comingNext.length ? (
          <p class="max-w-md text-[13px] text-muted sm:text-right">
            More open decision models are planned: {comingNext.join(', ')}.
          </p>
        ) : null}
      </div>
    </Section>
  )
}

const pillars: { icon: IconName; title: string; text: string }[] = [
  {
    icon: 'computer',
    title: 'Local',
    text: 'Runs on your machine with ONNX Runtime, on the CPU or an NVIDIA GPU. The server listens on 127.0.0.1 by default.',
  },
  {
    icon: 'code',
    title: 'Open weights',
    text: 'Weights come from their authors’ Hugging Face repositories, pinned to a commit and checked against sha256. Ollaya never re-hosts them, and the runtime is Apache-2.0.',
  },
  {
    icon: 'banknotes',
    title: 'No per-token fees',
    text: 'Run as many decisions as your hardware can handle. No metering and no API bill.',
  },
  {
    icon: 'adjustments',
    title: 'Calibrated',
    text: 'Probabilities you can put thresholds on. Each model ships its own calibration, and a Modelfile refits it on your labelled data.',
  },
]

function Private() {
  return (
    <Section
      id="private"
      title="Your data stays yours"
      lead="Private by default."
      body="Tickets, emails and user messages are often the most sensitive data you have. With Ollaya they are scored where they already live."
    >
      <ul class="grid gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-2" role="list">
        {pillars.map((p) => (
          <li class="bg-canvas p-6 md:p-8">
            <Icon name={p.icon} class="size-6 text-fg" />
            <h3 class="mt-4 text-base font-semibold text-fg">{p.title}</h3>
            <p class="mt-2 text-[15px] text-body">{p.text}</p>
          </li>
        ))}
      </ul>
    </Section>
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
    gpu: { ok: true, text: 'NVIDIA, CUDA 13 or 12', note: 'Command line' },
  },
  {
    name: 'Linux',
    detail: 'x86-64',
    app: { ok: true, text: 'Desktop app', note: 'AppImage, .deb, .rpm' },
    cli: { ok: true, text: 'Install script', note: 'systemd service' },
    gpu: { ok: true, text: 'NVIDIA, CUDA 13 or 12' },
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
    gpu: { ok: true, text: 'NVIDIA, CUDA 13 or 12' },
  },
  {
    name: 'Docker',
    detail: 'amd64 and arm64',
    app: { ok: false },
    cli: { ok: true, text: 'Image on GHCR' },
    gpu: { ok: true, text: 'NVIDIA, CUDA 13 or 12', note: ':cuda and :cuda12, amd64' },
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

function Platforms() {
  return (
    <Section
      id="platforms"
      title="Platforms"
      lead="Runs where you work."
      body="A desktop app and a command line for macOS, Windows and Linux, and a Docker image for servers. Every model runs on the CPU; an NVIDIA GPU on Linux, Windows, WSL 2 or Docker takes a request down to milliseconds."
    >
      <div class="overflow-hidden rounded-2xl border border-line">
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
      <div class="mt-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <a href="/download" class="inline-flex items-center gap-1.5 text-sm font-medium text-fg underline-offset-4 hover:underline">
          Install for your platform <Icon name="arrowRight" class="size-4" />
        </a>
        <p class="max-w-md text-[13px] text-muted sm:text-right">
          NVIDIA GPUs need driver R525 or newer; the install scripts fetch the CUDA libraries only when they find one.
          On a Mac, laya and nli run on the Apple GPU through MLX; other models, AMD and Intel GPUs, and the Windows and Linux desktop apps without the command line installed use the CPU.
        </p>
      </div>
    </Section>
  )
}

function Closer() {
  return (
    <section class="mx-auto mt-24 max-w-6xl px-4 text-center md:mt-36 md:px-6" aria-labelledby="closer-title">
      <LogoMark class="mx-auto size-20 text-fg" />
      <h2 id="closer-title" class="mt-6 text-3xl font-medium tracking-tight text-fg md:text-4xl">
        Get up and running in minutes.
      </h2>
      <p class="mx-auto mt-3 max-w-md text-body">
        One binary, one command: <code class="font-mono text-[0.9em] text-fg">ollaya run winnow:e4b</code>.
      </p>
      <div class="mt-8 flex justify-center">
        <a href="/download" class={btnPrimary}>
          <Icon name="download" class="size-4" />
          Download
        </a>
      </div>
      <p class="mt-4 text-[13px] text-muted">
        macOS, Windows, Linux and Docker · Apache-2.0 ·{' '}
        <a href={GITHUB_URL} class={textLink}>
          GitHub
        </a>
      </p>
    </section>
  )
}
