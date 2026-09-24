import type { Child } from 'hono/jsx'
import { Icon, type IconName } from '../components/Icon'
import { LogoMark } from '../components/Logo'
import { btnPrimary, Code, CodeBlock, textLink } from '../components/ui'
import { catalog, comingNext, featuredTags, fullName } from '../data/catalog'
import { GITHUB_URL, LOCAL_API } from '../site'

const sections = [
  { id: 'fast', label: 'Fast' },
  { id: 'compatible', label: 'Drop-in compatible' },
  { id: 'models', label: 'Open models' },
  { id: 'private', label: 'Your data stays yours' },
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
          <Fast />
          <Compatible />
          <OpenModels />
          <Private />
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
        </div>
        <TerminalMock />
      </div>
    </section>
  )
}

// Real output of the command shown, run on an RTX 4090 (fp16): laya routed the English text to
// laya:en, which answered the triage preset's five questions in 8.9 ms (`--verbose` timings).
// Two lines: the second hangs under the first, and keeps that indent when it wraps on a phone.
const MOCK_COMMAND = ['ollaya run laya --preset triage \\', '"I was charged twice this month and want a refund."']
const mockRows = [
  { q: 'intent', a: 'refund', p: 1.0 },
  { q: 'is_urgent', a: 'no', p: 0.87 },
  { q: 'frustration', a: '1.59 / 3', note: 'clearly annoyed', p: 0.36 },
  { q: 'refund_requested', a: 'yes', p: 0.88 },
  { q: 'churn_risk', a: 'no', p: 0.89 },
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
          <span class="absolute inset-x-0 text-center text-xs text-muted">ollaya — zsh</span>
        </div>
        <div class="p-4 font-mono text-[12.5px] leading-6 text-fg sm:p-5 sm:text-[13px]">
          <PromptLine>
            <pre class="whitespace-pre-wrap">
              <Code code={MOCK_COMMAND[0]!} lang="shell" />
            </pre>
            <pre class="pl-[2ch] whitespace-pre-wrap">
              <Code code={MOCK_COMMAND[1]!} lang="shell" />
            </pre>
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
                  <td class="w-[18ch] py-0.5 pr-4 align-middle text-muted">{r.q}</td>
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
          <div class="term-row mt-4" style="--row:5">
            <PromptLine>
              <span class="term-cursor inline-block h-[1.15em] w-[1ch] translate-y-[0.2em] bg-fg/80" aria-hidden="true"></span>
            </PromptLine>
          </div>
        </div>
      </div>
      <figcaption class="mt-3 text-center text-xs text-muted lg:absolute lg:inset-x-0 lg:top-full">
        Real output: routed to <span class="font-mono">laya:en</span>, answered in 8.9 ms on an RTX 4090.
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

// Median end-to-end latency of a five-question request through the HTTP API, on an RTX 4090 (fp16 for
// laya, fp32 for the others). Jev: the hosted API's median request in third-party benchmarks.
const latency = [
  { label: 'laya:multilingual', ms: 8.1, text: '8.1 ms' },
  { label: 'laya:en', ms: 9.6, text: '9.6 ms' },
  { label: 'gliclass', ms: 14.7, text: '14.7 ms' },
  { label: 'nli', ms: 20.4, text: '20.4 ms' },
  { label: 'decider:0.8b', ms: 155, text: '155 ms' },
  { label: 'decider:2b', ms: 190, text: '190 ms' },
]
const JEV = { label: 'TypeSafe Jev', note: 'hosted API', from: 236, to: 276, text: '236–276 ms' }
const LATENCY_SCALE = 300
const AXIS = [0, 100, 200, 300]

const pct = (n: number) => `${((n / LATENCY_SCALE) * 100).toFixed(2)}%`

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

function Fast() {
  return (
    <Section
      id="fast"
      title="Fast"
      lead="Decisions in milliseconds."
      body="A decision model answers in a single forward pass, with no token-by-token generation. On your own GPU, a five-question request to Laya takes about 10 ms, end to end through the HTTP API."
    >
      <div class="grid overflow-hidden rounded-2xl border border-line sm:grid-cols-2">
        <Stat value="8–10" unit="ms" label="Laya on Ollaya" detail="RTX 4090, five questions, end to end" />
        <div class="border-t border-line sm:border-t-0 sm:border-l">
          <Stat value="236–276" unit="ms" label="TypeSafe Jev" detail="Hosted API, median request" muted />
        </div>
      </div>

      <figure class="mt-14">
        <figcaption class="text-sm font-medium text-fg">
          Every model, one scale <span class="font-normal text-muted">· median latency, lower is better</span>
        </figcaption>
        <div class="mt-6 grid grid-cols-[8.25rem_minmax(0,1fr)] gap-x-3 sm:grid-cols-[9.5rem_minmax(0,1fr)] sm:gap-x-4">
          <ul class="contents" role="list">
            {latency.map((r) => (
              <li class="contents">
                <span class="flex h-9 items-center font-mono text-xs text-fg sm:text-[13px]">{r.label}</span>
                <span class="relative flex h-9 items-center">
                  <span class="lat-bar h-2.5 min-w-1 rounded-full" style={`width:${pct(r.ms)}`}></span>
                  <span class="ml-2.5 text-[13px] font-medium whitespace-nowrap text-fg tabular-nums">{r.text}</span>
                </span>
              </li>
            ))}
            <li class="contents">
              <span class="flex h-9 flex-col justify-center text-[13px] leading-tight">
                <span class="text-body">{JEV.label}</span>
                <span class="text-xs text-muted">{JEV.note}</span>
              </span>
              <span class="relative flex h-9 items-center">
                <span class="h-2.5 rounded-l-full bg-bar-muted" style={`width:${pct(JEV.from)}`}></span>
                <span class="h-2.5 rounded-r-full bg-bar-muted opacity-50" style={`width:${pct(JEV.to - JEV.from)}`}></span>
                <span class="ml-2.5 text-[13px] whitespace-nowrap text-body tabular-nums">{JEV.text}</span>
              </span>
            </li>
          </ul>
          <span></span>
          <span class="relative mt-2 h-5 border-t border-line text-[11px] text-muted tabular-nums" aria-hidden="true">
            {AXIS.map((t) => (
              <span
                class={`absolute top-1.5 whitespace-nowrap ${t === 0 ? '' : t === LATENCY_SCALE ? '-translate-x-full' : '-translate-x-1/2'}`}
                style={`left:${pct(t)}`}
              >
                {t === LATENCY_SCALE ? `${t} ms` : t}
              </span>
            ))}
          </span>
        </div>
        <p class="mt-8 max-w-2xl text-[13px] leading-relaxed text-muted">
          Ollaya: median of a five-question request through the HTTP API on an NVIDIA RTX 4090 (laya in fp16, the
          others in fp32). Jev: median request latency of the hosted API in third-party benchmarks (
          <a href="https://github.com/AbdelStark/jev-benchmarks" class={textLink}>
            AbdelStark/jev-benchmarks
          </a>
          ,{' '}
          <a href="https://github.com/nibzard/decision-model-benchmark" class={textLink}>
            nibzard/decision-model-benchmark
          </a>
          ), which includes the network. Setups differ, so read it as an order-of-magnitude comparison.
        </p>
      </figure>
    </Section>
  )
}

const compatRequest = `# Point the TypeSafe SDK at Ollaya
export TYPESAFE_BASE_URL=${LOCAL_API}
export TYPESAFE_API_KEY=local        # any value works
export TYPESAFE_DEFAULT_MODEL=laya

# …or call the compatible endpoint directly
curl ${LOCAL_API}/v1/systemone -d '{
    "model": "laya",
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
  "model": "laya:en",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "invoice",
      "confidence": 0.9547,
      "probabilities": {
        "invoice": 0.9698,
        "refund": 0.0172,
        "other": 0.013
      }
    }
  },
  "usage": {
    "input_tokens": 43,
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

/** One row per model, or — while the registry holds a single family — one row per featured tag. */
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
          ? 'Start with Laya from Convai Innovations: an English model, a 100+ language model, a model fine-tuned for typed decisions, and a router that picks for you.'
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
            Planned: more open decision models — {comingNext.join(', ')}.
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
    text: 'Probabilities you can put thresholds on. Laya’s calibration error (ECE) is 0.081 after temperature fitting, vs 0.246 for Jev.',
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

function Closer() {
  return (
    <section class="mx-auto mt-24 max-w-6xl px-4 text-center md:mt-36 md:px-6" aria-labelledby="closer-title">
      <LogoMark class="mx-auto size-20 text-fg" />
      <h2 id="closer-title" class="mt-6 text-3xl font-medium tracking-tight text-fg md:text-4xl">
        Get up and running in minutes.
      </h2>
      <p class="mx-auto mt-3 max-w-md text-body">
        One binary, one command: <code class="font-mono text-[0.9em] text-fg">ollaya run laya</code>.
      </p>
      <div class="mt-8 flex justify-center">
        <a href="/download" class={btnPrimary}>
          <Icon name="download" class="size-4" />
          Download
        </a>
      </div>
      <p class="mt-4 text-[13px] text-muted">
        Linux, macOS and Docker · Apache-2.0 ·{' '}
        <a href={GITHUB_URL} class={textLink}>
          GitHub
        </a>
      </p>
    </section>
  )
}
