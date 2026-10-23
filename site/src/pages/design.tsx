/**
 * /design: a demo of the site's next design, built from the same data as the live pages. Not in
 * the sitemap, noindex. Dark only for now; the tokens are scoped to .rd so app.css is untouched.
 */
import type { Child } from 'hono/jsx'
import { Scatter } from '../components/charts'
import { Icon } from '../components/Icon'
import { LogoMark } from '../components/Logo'
import { CodeBlock, CopyButton } from '../components/ui'
import { catalog } from '../data/catalog'
import { benchGpu, result } from '../data/numbers'
import stats from '../data/stats.json'
import { asset } from '../lib/assets'
import { benchUrl } from '../lib/bench'
import { githubStars } from '../lib/stars'
import { GITHUB_URL } from '../site'
import { compatRequest, compatResponse, JEV, OLLAMA, speedup, TerminalMock, vs, vsFeatures } from './home'
import { accuracyScatter } from './results'

const INSTALL = 'curl -fsSL https://ollaya.dev/install.sh | sh'
const lastWeek = stats.weeks.at(-1)!
const fmt = (n: number) => n.toLocaleString('en-US')
const mostAccurate = vs('pplx-decider:flash', 'ollaya')

/** Publishers worth a name on the page, in the order they read best. */
const PUBLISHERS = ['Perplexity AI', 'Cloudflare', 'PostHog', 'Bespoke Labs', 'Qwen', 'Strands Agents', 'Knowledgator', 'Convai Innovations']

const tokens = `
@font-face { font-family: 'Instrument Serif'; font-style: normal; font-weight: 400; font-display: swap; src: url(${asset('fonts/instrument-serif.woff2')}) format('woff2'); }
@font-face { font-family: 'Instrument Serif'; font-style: italic; font-weight: 400; font-display: swap; src: url(${asset('fonts/instrument-serif-italic.woff2')}) format('woff2'); }
.rd {
  color-scheme: dark;
  --color-canvas: #0b0c0b; --color-fg: #f4f3ef; --color-body: #c9c8c2; --color-muted: #8f8e88; --color-faint: #6b6a65;
  --color-line: rgb(255 255 255 / 0.08); --color-line-strong: rgb(255 255 255 / 0.16); --color-subtle: rgb(255 255 255 / 0.03);
  --color-code: rgb(255 255 255 / 0.04); --color-fill: rgb(255 255 255 / 0.06); --color-fill-strong: rgb(255 255 255 / 0.1);
  --color-btn: #f4f3ef; --color-btn-hover: #ffffff; --color-btn-fg: #0b0c0b; --color-term: rgb(12 13 12 / 0.72);
  --color-tok-comment: #7d8590; --color-tok-string: #9fe3b4; --color-tok-number: #d2a8ff; --color-tok-keyword: #ff9a9a;
  --color-tok-key: #8cc8ff; --color-tok-fn: #ffd080; --color-tok-punct: #8b949e;
  --color-us: #9fd7ff; --color-them: #f0a060; --color-pass: #3fb950; --color-fail: #f85149;
  --serif: 'Instrument Serif', 'Iowan Old Style', 'Times New Roman', serif;
  background: var(--color-canvas); color: var(--color-body);
}
.rd .serif { font-family: var(--serif); font-weight: 400; letter-spacing: -0.01em; }
.rd .glass {
  background: linear-gradient(180deg, rgb(255 255 255 / 0.07), rgb(255 255 255 / 0.03));
  border: 1px solid rgb(255 255 255 / 0.1);
  box-shadow: 0 1px 0 0 rgb(255 255 255 / 0.08) inset, 0 30px 80px -40px rgb(0 0 0 / 0.8);
  backdrop-filter: blur(18px) saturate(1.2); -webkit-backdrop-filter: blur(18px) saturate(1.2);
}
.rd .glass-strong { background: rgb(14 15 14 / 0.72); }
/* The grain: an SVG noise tile over everything, blended so it only breaks up the flat colour. */
.rd .grain { position: fixed; inset: 0; z-index: 50; pointer-events: none; opacity: 0.09; mix-blend-mode: soft-light;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size: 240px 240px; }
/* Blurred colour fields behind the hero and the closer: the backdrop the glass sits on. */
.rd .field { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
.rd .field i { position: absolute; display: block; border-radius: 9999px; filter: blur(90px); opacity: 0.55; }
.rd .field .a { width: 60vw; height: 60vw; left: -10vw; top: -20vw; background: #2d5a4a; }
.rd .field .b { width: 50vw; height: 50vw; right: -12vw; top: -8vw; background: #5a4a2d; }
.rd .field .c { width: 45vw; height: 45vw; left: 30vw; top: 20vw; background: #2d3f5a; opacity: 0.4; }
@media (prefers-reduced-motion: no-preference) {
  .rd .field i { animation: drift 22s ease-in-out infinite alternate; }
  .rd .field .b { animation-duration: 28s; animation-delay: -9s; }
  .rd .field .c { animation-duration: 34s; animation-delay: -17s; }
}
@keyframes drift { to { transform: translate3d(6vw, 4vw, 0) scale(1.08); } }
.rd .term-glow { box-shadow: 0 1px 0 0 rgb(255 255 255 / 0.08) inset, 0 40px 120px -40px rgb(0 0 0 / 0.9); }
.rd .hairline { background: linear-gradient(90deg, transparent, rgb(255 255 255 / 0.18), transparent); height: 1px; }
.rd ::selection { background: rgb(159 215 255 / 0.3); }
`

function Nav() {
  const stars = githubStars()
  const links = [
    ['/search', 'Models'],
    [benchUrl(), 'Bench'],
    ['/docs', 'Docs'],
    ['/blog', 'Blog'],
  ]
  return (
    <header class="fixed inset-x-0 top-0 z-40 px-4 pt-4 md:px-6">
      <nav aria-label="Main" class="glass glass-strong mx-auto flex h-14 max-w-6xl items-center gap-6 rounded-full px-5 pr-2">
        <a href="/" class="flex items-center gap-2 text-fg" aria-label="Ollaya home">
          <LogoMark class="size-7" />
          <span class="text-[15px] font-medium tracking-tight">ollaya</span>
        </a>
        <ul class="hidden items-center gap-5 text-sm text-body md:flex">
          {links.map(([href, label]) => (
            <li>
              <a href={href} class="hover:text-fg">
                {label}
              </a>
            </li>
          ))}
        </ul>
        <div class="ml-auto flex items-center gap-2">
          <a href={GITHUB_URL} class="hidden items-center gap-1.5 rounded-full px-3 py-1.5 text-sm text-body hover:text-fg sm:inline-flex">
            <Icon name="star" class="size-3.5" />
            {stars ?? 'GitHub'}
          </a>
          <a href="/download" class="inline-flex h-10 items-center rounded-full bg-btn px-4 text-sm font-medium text-btn-fg hover:bg-btn-hover">
            Download
          </a>
        </div>
      </nav>
    </header>
  )
}

function Install() {
  return (
    <div class="glass glass-strong inline-flex max-w-full items-center gap-3 rounded-full py-2 pr-2 pl-5 font-mono text-[13px] text-fg" data-copy-scope>
      <span class="text-muted select-none" aria-hidden="true">
        $
      </span>
      <pre class="truncate">{INSTALL}</pre>
      <CopyButton class="rounded-full bg-fill" />
    </div>
  )
}

function Hero() {
  return (
    <section class="relative overflow-hidden pt-40 pb-16 md:pt-48" aria-labelledby="rd-hero">
      <div class="field" aria-hidden="true">
        <i class="a"></i>
        <i class="b"></i>
        <i class="c"></i>
      </div>
      <div class="relative mx-auto max-w-6xl px-4 text-center md:px-6">
        <p class="inline-flex items-center gap-2 rounded-full border border-line px-3 py-1 text-[13px] text-body">
          <span class="size-1.5 rounded-full bg-pass"></span>
          Open source · Apache-2.0 · {catalog.length} open decision models
        </p>
        <h1 id="rd-hero" class="serif mx-auto mt-6 max-w-4xl text-[3.25rem] leading-[0.98] text-fg md:text-[5.5rem]">
          The decision layer <br class="hidden md:block" />
          for AI agents.
        </h1>
        <p class="mx-auto mt-6 max-w-2xl text-lg text-body md:text-xl">
          Ollaya runs open decision models on your own hardware. Typed questions in, calibrated answers out, in
          milliseconds: routing, guardrails, triage and tool approval without an LLM call.
        </p>
        <div class="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Install />
          <a href="/try" class="inline-flex h-11 items-center gap-1.5 rounded-full px-4 text-sm font-medium text-fg hover:underline underline-offset-4">
            Try it in the browser <Icon name="arrowRight" class="size-4" />
          </a>
        </div>
      </div>
      <div class="relative mx-auto mt-16 max-w-3xl px-4 md:px-6">
        <TerminalMock />
      </div>
    </section>
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
    <section class="mx-auto max-w-6xl px-4 md:px-6" aria-label="In numbers">
      <div class="hairline"></div>
      <dl class="grid grid-cols-2 gap-y-8 py-10 md:grid-cols-4">
        {items.map(([v, l]) => (
          <div class="text-center">
            <dd class="serif text-4xl text-fg tabular-nums md:text-5xl">{v}</dd>
            <dt class="mt-1 text-sm text-muted">{l}</dt>
          </div>
        ))}
      </dl>
      <div class="hairline"></div>
      <p class="mt-8 text-center text-xs tracking-[0.18em] text-muted uppercase">Models by</p>
      <ul class="mt-4 flex flex-wrap items-center justify-center gap-x-8 gap-y-3 text-[15px] font-medium text-body" role="list">
        {PUBLISHERS.map((p) => (
          <li>{p}</li>
        ))}
      </ul>
    </section>
  )
}

function Title({ kicker, children, lead }: { kicker: string; children: Child; lead?: Child }) {
  return (
    <div class="mx-auto max-w-2xl text-center">
      <p class="text-xs tracking-[0.18em] text-muted uppercase">{kicker}</p>
      <h2 class="serif mt-3 text-4xl leading-[1.02] text-fg md:text-6xl">{children}</h2>
      {lead ? <p class="mt-5 text-base text-body md:text-lg">{lead}</p> : null}
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
    <section class="mx-auto mt-32 max-w-6xl px-4 md:px-6" aria-labelledby="rd-why">
      <Title kicker="Why decision models" lead="A decision model answers typed questions about a text or a JSON: a choice, a score, a yes or no. It is the fast, reliable half of an agent.">
        <span id="rd-why">System 1 for your agents.</span>
      </Title>
      <ul class="mt-14 grid gap-4 md:grid-cols-3" role="list">
        {why.map((w) => (
          <li class="glass rounded-3xl p-7">
            <Icon name={w.icon} class="size-6 text-fg" />
            <h3 class="mt-5 text-lg font-medium text-fg">{w.title}</h3>
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

function Compare() {
  const chart = accuracyScatter()
  return (
    <section class="mx-auto mt-32 max-w-6xl px-4 md:px-6" aria-labelledby="rd-compare">
      <Title
        kicker="Measured"
        lead={`${OLLAMA}'s nine decision models and Ollaya's, on the same ${benchGpu} and Bespoke Labs' 3,880 human-labeled questions. Ollaya runs the most accurate model and more of them, with each author's temperature. On the same weights Ollama is ${speedup()} times faster: it runs Q8_0 GGUFs where Ollaya computes the authors' BF16 weights.`}
      >
        <span id="rd-compare">Against Ollama and hosted Jev.</span>
      </Title>
      <div class="glass mt-14 overflow-hidden rounded-3xl">
        <table class="w-full border-collapse text-[15px]">
          <thead>
            <tr class="border-b border-line text-left text-[13px]">
              <th class="px-6 py-4 font-medium text-muted"></th>
              <th class="px-6 py-4 font-medium text-fg">Ollaya</th>
              <th class="px-6 py-4 font-medium text-muted">{OLLAMA}</th>
              <th class="hidden px-6 py-4 font-medium text-muted md:table-cell">TypeSafe Jev, hosted</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-line">
            {rows.map((r) => (
              <tr>
                <th scope="row" class="px-6 py-3.5 text-left font-medium text-fg">
                  {r.label}
                </th>
                <td class="px-6 py-3.5 align-top">
                  <CellView c={r.ollaya} strong />
                </td>
                <td class="px-6 py-3.5 align-top">
                  <CellView c={r.ollama} />
                </td>
                <td class="hidden px-6 py-3.5 align-top md:table-cell">
                  <CellView c={r.jev} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figure class="glass mt-6 rounded-3xl p-6 md:p-8">
        <figcaption class="text-sm text-body">
          <span class="font-medium text-fg">Accuracy against latency</span>, the most accurate tag of each model on each server
        </figcaption>
        <div class="mt-4 hidden md:block">
          <Scatter {...chart} />
        </div>
        <div class="mt-4 md:hidden">
          <Scatter {...chart} compact />
        </div>
      </figure>
      <p class="mt-6 text-center text-sm">
        <a href={benchUrl()} class="inline-flex items-center gap-1.5 font-medium text-fg underline-offset-4 hover:underline">
          Every model on Ollaya Bench, with the raw data <Icon name="arrowRight" class="size-4" />
        </a>
      </p>
    </section>
  )
}

function Api() {
  return (
    <section class="mx-auto mt-32 max-w-6xl px-4 md:px-6" aria-labelledby="rd-api">
      <Title kicker="Drop-in" lead="Ollaya serves TypeSafe's request and response shapes. Point the official SDK at localhost and nothing else changes.">
        <span id="rd-api">Jev-compatible, self-hosted.</span>
      </Title>
      <div class="mt-14 grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        {[
          ['Request', compatRequest, 'shell'],
          ['Response', compatResponse, 'json'],
        ].map(([label, code, lang]) => (
          <div class="glass glass-strong flex flex-col overflow-hidden rounded-3xl">
            <div class="flex items-center gap-2 border-b border-line px-5 py-3" aria-hidden="true">
              <span class="flex gap-1.5">
                <span class="size-2.5 rounded-full bg-fill-strong"></span>
                <span class="size-2.5 rounded-full bg-fill-strong"></span>
                <span class="size-2.5 rounded-full bg-fill-strong"></span>
              </span>
              <span class="ml-2 text-xs text-muted">{label}</span>
            </div>
            <CodeBlock code={code!} lang={lang} class="flex-1 [&>pre]:rounded-none [&>pre]:bg-transparent" />
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
    <a href={`/library/${m.name}`} class="glass group flex flex-col rounded-3xl p-6 transition-colors hover:border-line-strong">
      <div class="flex items-baseline justify-between gap-3">
        <span class="font-mono text-[17px] text-fg group-hover:underline underline-offset-4">{m.name}</span>
        <span class="truncate text-xs text-muted">{m.publisher?.name ?? ''}</span>
      </div>
      <p class="mt-3 line-clamp-3 flex-1 text-[14px] text-body">{summary}</p>
      <dl class="mt-5 flex items-end justify-between gap-4 border-t border-line pt-4">
        <div>
          <dd class="serif text-3xl text-fg tabular-nums">{m.stats?.accuracy ? m.stats.accuracy.toFixed(3) : '—'}</dd>
          <dt class="text-xs text-muted">accuracy</dt>
        </div>
        <div>
          <dd class="serif text-3xl text-fg tabular-nums">{m.stats?.latencyMs ? `${Math.round(m.stats.latencyMs)} ms` : '—'}</dd>
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
    <section class="mx-auto mt-32 max-w-6xl px-4 md:px-6" aria-labelledby="rd-models">
      <Title kicker="Open models" lead={`${catalog.length} families from their authors, pulled by name. Every number here is from Ollaya Bench on an ${benchGpu}.`}>
        <span id="rd-models">Open weights, ready to pull.</span>
      </Title>
      <ul class="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-3" role="list">
        {top.map((m) => (
          <li class="flex">
            <ModelCard m={m} />
          </li>
        ))}
      </ul>
      <p class="mt-8 text-center text-sm">
        <a href="/search" class="inline-flex items-center gap-1.5 font-medium text-fg underline-offset-4 hover:underline">
          Browse all {catalog.length} models <Icon name="arrowRight" class="size-4" />
        </a>
      </p>
    </section>
  )
}

function Closer() {
  return (
    <section class="relative mt-32 overflow-hidden py-32 text-center" aria-labelledby="rd-closer">
      <div class="field" aria-hidden="true">
        <i class="a" style="top:auto;bottom:-30vw"></i>
        <i class="b" style="top:auto;bottom:-20vw"></i>
      </div>
      <div class="relative mx-auto max-w-6xl px-4 md:px-6">
        <LogoMark class="mx-auto size-14 text-fg" />
        <h2 id="rd-closer" class="serif mt-6 text-5xl leading-none text-fg md:text-7xl">
          Up and running <em>in a minute.</em>
        </h2>
        <p class="mx-auto mt-5 max-w-md text-body">macOS, Windows, Linux and Docker. CPU, NVIDIA and Apple GPUs.</p>
        <div class="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Install />
          <a href="/download" class="inline-flex h-11 items-center gap-2 rounded-full bg-btn px-5 text-sm font-medium text-btn-fg hover:bg-btn-hover">
            <Icon name="download" class="size-4" />
            Download
          </a>
        </div>
      </div>
    </section>
  )
}

function Foot() {
  return (
    <footer class="mx-auto flex max-w-6xl flex-col gap-3 px-4 pb-10 text-[13px] text-muted sm:flex-row sm:items-center sm:justify-between md:px-6">
      <p>© 2026 Ollaya · Apache-2.0 · An independent project, not affiliated with Ollama or TypeSafe.</p>
      <p class="flex gap-5">
        <a href={GITHUB_URL} class="hover:text-fg">
          GitHub
        </a>
        <a href="/docs" class="hover:text-fg">
          Docs
        </a>
        <a href="/contributors" class="hover:text-fg">
          Contributors
        </a>
      </p>
    </footer>
  )
}

export function DesignPage({ origin }: { origin: string }) {
  return (
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Design demo · Ollaya</title>
        <meta name="robots" content="noindex" />
        <meta name="theme-color" content="#0b0c0b" />
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <link rel="stylesheet" href={asset('app.css')} />
        <style dangerouslySetInnerHTML={{ __html: tokens }}></style>
        <script src={asset('app.js')} defer></script>
      </head>
      <body class="rd min-h-screen font-sans antialiased" data-origin={origin}>
        <div class="grain" aria-hidden="true"></div>
        <Nav />
        <main>
          <Hero />
          <Proof />
          <Why />
          <Compare />
          <Api />
          <Models />
          <Closer />
        </main>
        <Foot />
      </body>
    </html>
  )
}
