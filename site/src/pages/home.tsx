import type { Child } from 'hono/jsx'
import { Icon, type IconName } from '../components/Icon'
import { LogoMark } from '../components/Logo'
import { btnPrimary, CodeBlock, textLink } from '../components/ui'
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
          <ul class="sticky top-28 space-y-2.5 text-sm" data-scrollspy>
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

const mockRows = [
  { q: 'department', a: 'billing', p: 0.85, w: 'w-[85%]' },
  { q: 'urgency', a: 'normal · 1.20', p: 0.56, w: 'w-[56%]' },
  { q: 'refund', a: 'yes', p: 0.91, w: 'w-[91%]' },
]

function TerminalMock() {
  return (
    // The caption sits outside the flow on wide screens, so the hero text centres on the terminal
    // window itself rather than on window + caption.
    <figure class="relative min-w-0">
      <div class="overflow-hidden rounded-xl border border-line bg-term shadow-2xl shadow-black/5">
        <div class="flex items-center gap-1.5 border-b border-line px-4 py-3" aria-hidden="true">
          <span class="size-2.5 rounded-full bg-line-strong"></span>
          <span class="size-2.5 rounded-full bg-line-strong"></span>
          <span class="size-2.5 rounded-full bg-line-strong"></span>
        </div>
        <div class="p-4 font-mono text-[13px] leading-6 text-fg sm:p-5">
          <p class="break-words">
            <span class="text-muted select-none">$ </span>ollaya run laya --preset triage "I was charged twice for my
            subscription this month…"
          </p>
          <p class="text-muted">routed to laya:en (English)</p>
          <table class="mt-4 w-full border-collapse text-left">
            <caption class="sr-only">Answers returned by the model</caption>
            <thead>
              <tr class="text-[11px] tracking-wider text-muted uppercase">
                <th scope="col" class="pr-4 pb-1 font-normal">Question</th>
                <th scope="col" class="pr-4 pb-1 font-normal">Answer</th>
                <th scope="col" class="pb-1 font-normal">
                  <span class="hidden sm:inline">Probability</span>
                  <span class="sm:hidden">P</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {mockRows.map((r) => (
                <tr>
                  <td class="py-0.5 pr-4 text-muted">{r.q}</td>
                  <td class="py-0.5 pr-4 whitespace-nowrap">{r.a}</td>
                  <td class="py-0.5">
                    <span class="flex items-center gap-2.5">
                      <span class="hidden h-1.5 w-20 overflow-hidden rounded-full bg-fill-strong min-[380px]:block" aria-hidden="true">
                        <span class={`block h-full rounded-full bg-bar ${r.w}`}></span>
                      </span>
                      <span class="tabular-nums">{r.p.toFixed(2)}</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p class="mt-4 text-muted">
            <span class="select-none">$ </span>
            <span class="inline-block h-4 w-2 translate-y-0.5 bg-fg/70" aria-hidden="true"></span>
          </p>
        </div>
      </div>
      <figcaption class="mt-3 text-[13px] text-muted lg:absolute lg:top-full lg:left-0">
        Illustrative output — the CLI is being built and may change.
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

const latency = [
  { label: 'Laya multilingual', from: 32.8, to: 32.8, text: '32.8 ms', ours: true },
  { label: 'Laya', from: 39.5, to: 39.5, text: '39.5 ms', ours: true },
  { label: 'TypeSafe Jev (p50)', from: 236, to: 276, text: '236–276 ms', ours: false },
]
const LATENCY_MAX = 276

const pct = (n: number) => `${((n / LATENCY_MAX) * 100).toFixed(1)}%`

function Fast() {
  return (
    <Section
      id="fast"
      title="Fast"
      lead="Decisions in tens of milliseconds."
      body="A decision model answers in a single forward pass. There is no token-by-token generation, so one question takes under 40 ms on a single NVIDIA T4."
    >
      <figure>
        <figcaption class="text-sm font-medium text-fg">
          Latency for one question <span class="font-normal text-muted">· lower is better</span>
        </figcaption>
        <ul class="mt-5 space-y-5" role="list">
          {latency.map((r) => (
            <li
              class="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-2 sm:grid-cols-[10rem_minmax(0,1fr)_6.5rem]"
              title={`${r.label}: ${r.text}`}
            >
              <span class={`text-sm ${r.ours ? 'font-medium text-fg' : 'text-body'}`}>{r.label}</span>
              <span class="text-right text-sm text-fg tabular-nums sm:order-last">{r.text}</span>
              <span class="col-span-2 flex h-2.5 sm:col-span-1" aria-hidden="true">
                <span
                  class={`h-full rounded-r-[4px] ${r.ours ? 'bg-bar' : 'bg-bar-muted'}`}
                  style={`width:${pct(r.from)}`}
                ></span>
                {r.to > r.from ? (
                  <span
                    class="ml-[2px] h-full rounded-r-[4px] bg-bar-muted opacity-50"
                    style={`width:calc(${pct(r.to - r.from)} - 2px)`}
                  ></span>
                ) : null}
              </span>
            </li>
          ))}
        </ul>
        <p class="mt-6 max-w-2xl text-[13px] leading-relaxed text-muted">
          Laya figures are from the{' '}
          <a href="https://huggingface.co/convaiinnovations/laya" class={textLink}>
            Laya model card
          </a>
          , measured on an NVIDIA Tesla T4. Jev p50 range from third-party benchmarks (
          <a href="https://github.com/AbdelStark/jev-benchmarks" class={textLink}>
            AbdelStark/jev-benchmarks
          </a>
          ,{' '}
          <a href="https://github.com/nibzard/decision-model-benchmark" class={textLink}>
            nibzard/decision-model-benchmark
          </a>
          ). Setups differ, so treat this as an order-of-magnitude comparison.
        </p>
      </figure>
    </Section>
  )
}

const compatRequest = `# Point an existing TypeSafe SDK at your local server
export TYPESAFE_BASE_URL=${LOCAL_API}

# …or call the compatible endpoint directly
curl ${LOCAL_API}/v1/systemone \\
  -H "Content-Type: application/json" \\
  -d '{
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
  "model": "laya",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "invoice",
      "confidence": 0.9,
      "probabilities": {
        "invoice": 0.933,
        "refund": 0.021,
        "other": 0.046
      }
    }
  },
  "usage": {
    "input_tokens": 52,
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
          <code class="font-mono text-[0.9em]">/v1/models</code> with the same request and response shapes, so
          existing TypeSafe SDKs work by changing one environment variable.
        </>
      }
    >
      <div class="grid grid-cols-[minmax(0,1fr)] gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div>
          <p class="mb-2 text-[13px] font-medium text-muted">Request</p>
          <CodeBlock code={compatRequest} />
        </div>
        <div>
          <p class="mb-2 text-[13px] font-medium text-muted">Response</p>
          <CodeBlock code={compatResponse} />
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

function OpenModels() {
  const laya = catalog[0]!
  return (
    <Section
      id="models"
      title="Open models"
      lead="Open weights, ready to pull."
      body="Start with Laya from Convai Innovations: an English model, a 100+ language model, a model fine-tuned for typed decisions, and a router that picks for you."
    >
      <ul class="divide-y divide-line border-y border-line" role="list">
        {featuredTags(laya).map((t) => (
          <li>
            <a
              href={`/library/${fullName(laya, t)}`}
              class="group flex flex-col gap-1 py-5 sm:flex-row sm:items-baseline sm:gap-6"
            >
              <span class="shrink-0 font-mono text-[15px] text-fg underline-offset-4 group-hover:underline sm:w-52">
                {fullName(laya, t)}
              </span>
              <span class="flex-1 text-body">{t.summary}</span>
              <span class="text-[13px] whitespace-nowrap text-muted">
                {t.kind === 'router' ? 'router' : `${t.params} · ${t.context} ctx`}
              </span>
            </a>
          </li>
        ))}
      </ul>
      <div class="mt-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <a href="/search" class="inline-flex items-center gap-1.5 text-sm font-medium text-fg underline-offset-4 hover:underline">
          Browse all models <Icon name="arrowRight" class="size-4" />
        </a>
        <p class="max-w-md text-[13px] text-muted sm:text-right">
          Coming next: more open decision models — {comingNext.join(', ')}.
        </p>
      </div>
    </Section>
  )
}

const pillars: { icon: IconName; title: string; text: string }[] = [
  {
    icon: 'computer',
    title: 'Local',
    text: 'Runs on your machine with ONNX Runtime — CUDA, Core ML or plain CPU. The server listens on 127.0.0.1 by default.',
  },
  {
    icon: 'code',
    title: 'Open weights',
    text: 'Apache-2.0 models you can inspect, fine-tune and redistribute. The runtime is Apache-2.0 too.',
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
        Pre-release —{' '}
        <a href={GITHUB_URL} class={textLink}>
          watch the repository
        </a>{' '}
        for the first version.
      </p>
    </section>
  )
}
