/**
 * Ollaya Bench's frame (header, footer, page shell) and the small pieces its pages share. It uses
 * the main site's tokens and stylesheet, so the two sites read as one family.
 */
import type { Child } from 'hono/jsx'
import { LogoMark } from '../components/Logo'
import { asset, hasAsset } from '../lib/assets'
import { ANALYTICS, GITHUB_URL } from '../site'
import { edition, fmtMb, fmtMs, modelUrl, rejectedShare, type Row } from './data'

export const BENCH_NAME = 'Ollaya Bench'
export const BENCH_TAGLINE = 'Every open decision model, measured on real hardware.'
export const BENCH_DESCRIPTION =
  'Accuracy, calibration, speed and memory of every open decision model, measured on GPUs and CPUs people own. Every number comes with its raw data and the command that reproduces it.'
export const BENCH_REPO_URL = `${GITHUB_URL}/tree/main/results`

export type BenchNav = 'leaderboard' | 'methodology' | 'data'

export interface BenchMeta {
  title?: string
  description?: string
  nav?: BenchNav
  noindex?: boolean
  /** Social card under /static, made by scripts/og-bench.mjs; pages without one use og-bench.png. */
  ogImage?: string
  ogAlt?: string
}

const links: { key: BenchNav; href: string; label: string }[] = [
  { key: 'leaderboard', href: '/', label: 'Leaderboard' },
  { key: 'methodology', href: '/methodology', label: 'Methodology' },
  { key: 'data', href: '/data', label: 'Data' },
]

function Header({ active, main }: { active?: BenchNav; main: string }) {
  return (
    <header class="fixed inset-x-0 top-0 z-40 border-b border-line bg-canvas/70 backdrop-blur-xl">
      <nav aria-label="Main" class="mx-auto flex h-14 w-full max-w-6xl items-center gap-5 px-5 md:gap-6">
        <a href="/" class="flex shrink-0 items-center gap-2 rounded-md text-fg" aria-label="Ollaya Bench home">
          <LogoMark class="size-7" />
          <span class="text-[17px] font-semibold tracking-[-0.035em]">
            ollaya <span class="font-medium text-muted">bench</span>
          </span>
        </a>
        <ul class="flex items-baseline gap-5 text-sm">
          {links.map((l) => (
            <li class={l.key === 'leaderboard' ? 'hidden sm:block' : undefined}>
              <a
                href={l.href}
                class={`underline-offset-4 hover:text-fg hover:underline ${active === l.key ? 'font-medium text-fg' : 'text-body'}`}
                aria-current={active === l.key ? 'page' : undefined}
              >
                {l.label}
              </a>
            </li>
          ))}
        </ul>
        <div class="ml-auto flex items-center gap-4 text-sm">
          {edition ? (
            <span class="hidden rounded-md border border-line px-2.5 py-0.5 font-mono text-xs text-muted md:inline">Edition {edition.id}</span>
          ) : null}
          <a href={main} class="hidden text-body underline-offset-4 hover:text-fg hover:underline sm:inline">
            {main.replace(/^https?:\/\//, '')} <span aria-hidden="true">↗</span>
          </a>
        </div>
      </nav>
    </header>
  )
}

function Footer({ main }: { main: string }) {
  return (
    <footer class="frame">
      <div class="rule flex flex-col gap-4 px-6 py-8 text-[13px] text-muted sm:flex-row sm:items-center sm:justify-between md:px-10">
        <p class="max-w-xl">
          Ollaya Bench is run by{' '}
          <a href={main} class="text-body underline-offset-4 hover:text-fg hover:underline">
            Ollaya
          </a>
          , the open source runtime for decision models. Every number here is measured on our own machines with the
          released Ollaya and published with its raw data.
        </p>
        <ul class="flex flex-wrap gap-x-5 gap-y-2">
          <li>
            <a href="/methodology" class="underline-offset-4 hover:text-fg hover:underline">
              Methodology
            </a>
          </li>
          <li>
            <a href="/data" class="underline-offset-4 hover:text-fg hover:underline">
              Data
            </a>
          </li>
          <li>
            <a href={GITHUB_URL} class="underline-offset-4 hover:text-fg hover:underline">
              GitHub
            </a>
          </li>
          <li>
            <a href={`${main}/search`} class="underline-offset-4 hover:text-fg hover:underline">
              Models
            </a>
          </li>
        </ul>
      </div>
    </footer>
  )
}

export function BenchLayout({
  origin,
  main,
  path,
  meta,
  children,
}: {
  origin: string
  main: string
  path: string
  meta: BenchMeta
  children: Child
}) {
  const title = meta.title ? `${meta.title} · ${BENCH_NAME}` : `${BENCH_NAME} · ${BENCH_TAGLINE.replace(/\.$/, '')}`
  const description = meta.description ?? BENCH_DESCRIPTION
  const canonical = `${origin}${path}`
  const card = meta.ogImage && hasAsset(meta.ogImage) ? meta.ogImage : 'og-bench.png'
  const ogImage = hasAsset(card) ? `${origin}${asset(card)}` : null
  const ogAlt = card === meta.ogImage && meta.ogAlt ? meta.ogAlt : `${BENCH_NAME}: ${BENCH_TAGLINE}`
  return (
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{title}</title>
        <meta name="description" content={description} />
        {meta.noindex ? <meta name="robots" content="noindex" /> : <link rel="canonical" href={canonical} />}
        <meta name="color-scheme" content="dark" />
        <meta name="theme-color" content="#0a0a0a" />
        <meta property="og:site_name" content={BENCH_NAME} />
        <meta property="og:type" content="website" />
        <meta property="og:title" content={meta.title ? `${meta.title} · ${BENCH_NAME}` : `${BENCH_NAME} · ${BENCH_TAGLINE}`} />
        <meta property="og:description" content={description} />
        <meta property="og:url" content={canonical} />
        {ogImage ? (
          <>
            <meta property="og:image" content={ogImage} />
            <meta property="og:image:width" content="1200" />
            <meta property="og:image:height" content="630" />
            <meta property="og:image:alt" content={ogAlt} />
            <meta name="twitter:card" content="summary_large_image" />
            <meta name="twitter:image" content={ogImage} />
          </>
        ) : null}
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        {ANALYTICS.key ? (
          <script async src={ANALYTICS.src} data-key={ANALYTICS.key} data-collector={ANALYTICS.collector}></script>
        ) : null}
        <link rel="stylesheet" href={asset('app.css')} />
        <script src={asset('bench.js')} defer></script>
      </head>
      <body class="flex min-h-screen flex-col bg-canvas font-sans text-body antialiased">
        <div class="grain" aria-hidden="true"></div>
        <a
          href="#main"
          class="sr-only rounded-md bg-btn px-4 py-2 text-sm text-btn-fg focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-[60]"
        >
          Skip to content
        </a>
        <Header active={meta.nav} main={main} />
        <main id="main" class="frame flex-1 border-b border-line pt-14 pb-16">
          {children}
        </main>
        <Footer main={main} />
      </body>
    </html>
  )
}

/** A section with an anchor, a title and an optional lead. */
export function Section({ id, title, lead, children, wide }: { id: string; title: string; lead?: Child; children: Child; wide?: boolean }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} class={`mx-auto mt-16 px-6 md:px-10 ${wide ? 'max-w-6xl' : 'max-w-4xl'}`}>
      <h2 id={`${id}-title`} class="text-2xl font-semibold tracking-[-0.02em] text-fg">
        {title}
      </h2>
      {lead ? <p class="mt-2 max-w-3xl text-[15px] leading-relaxed text-muted">{lead}</p> : null}
      <div class="mt-6">{children}</div>
    </section>
  )
}

export function ModelLink({ name, class: cls = '' }: { name: string; class?: string }) {
  return (
    <a href={modelUrl(name)} class={`font-mono whitespace-nowrap text-fg underline-offset-4 hover:underline ${cls}`}>
      {name}
    </a>
  )
}

/** A small label next to a model: why a number needs a second look. */
export function Flag({ children, title }: { children: Child; title: string }) {
  return (
    <span title={title} class="ml-1.5 inline-flex rounded-md bg-fill px-1.5 py-px align-[1px] font-sans text-[10.5px] font-medium text-muted">
      {children}
    </span>
  )
}

export function rowFlags(r: Row): Child[] {
  const out: Child[] = []
  if (r.score?.trained_on.length)
    out.push(
      <Flag title={`Trained on the train split of ${r.score.trained_on.join(', ')}, so its result there is not a held-out measurement: the score leaves it out.`}>
        trained on split
      </Flag>,
    )
  const rejected = rejectedShare(r)
  if (rejected >= 0.05)
    out.push(
      <Flag title="Share of the questions this model rejected (more options than it takes, a score scale it can't read, or an input longer than its context). A rejected question counts as wrong.">
        rejects {Math.round(rejected * 100)} %
      </Flag>,
    )
  if (r.router) out.push(<Flag title="A router: each request goes to laya:en or laya:multilingual by its language.">router</Flag>)
  if (r.builtin) out.push(<Flag title="Answers only its own built-in questions, so it has speed but no accuracy here.">built-in questions</Flag>)
  return out
}

/** The score with its 95 % interval, as a bar with a whisker on a 0..max scale. */
export function ScoreBar({ score, ci, max, accent }: { score: number; ci: [number, number]; max: number; accent?: boolean }) {
  const pct = (v: number) => `${Math.max(0, Math.min(100, (v / max) * 100)).toFixed(2)}%`
  return (
    <span class="inline-flex items-center gap-2.5" title={`95 % interval ${ci[0].toFixed(1)} to ${ci[1].toFixed(1)}`}>
      <span class="relative h-2 w-24 rounded-md bg-fill" aria-hidden="true">
        <span class={`absolute inset-y-0 left-0 rounded-md ${accent ? 'bg-us' : 'bg-bar'}`} style={`width:${pct(score)}`}></span>
        <span class="absolute top-1/2 h-px -translate-y-1/2 bg-muted" style={`left:${pct(ci[0])};width:calc(${pct(ci[1])} - ${pct(ci[0])})`}></span>
        <span class="absolute top-[-2px] bottom-[-2px] w-px bg-muted" style={`left:${pct(ci[0])}`}></span>
        <span class="absolute top-[-2px] bottom-[-2px] w-px bg-muted" style={`left:${pct(ci[1])}`}></span>
      </span>
      <span class="font-medium text-fg tabular-nums">{score.toFixed(1)}</span>
    </span>
  )
}

export const pct1 = (v: number | null | undefined) => (v === null || v === undefined ? '–' : `${(v * 100).toFixed(1)}`)
export const num3 = (v: number | null | undefined) => (v === null || v === undefined ? '–' : v.toFixed(3))
export { fmtMb, fmtMs }

/** A labelled figure: "Score 52.3". */
export function Stat({ label, value, detail }: { label: string; value: Child; detail?: Child }) {
  return (
    <div class="min-w-0">
      <dt class="text-xs text-muted">{label}</dt>
      <dd class="mt-1 text-2xl font-semibold tracking-tight text-fg tabular-nums">{value}</dd>
      {detail ? <dd class="mt-0.5 text-xs text-muted">{detail}</dd> : null}
    </div>
  )
}

export function Mono({ children }: { children: Child }) {
  return <code class="rounded bg-code px-1.5 py-0.5 font-mono text-[0.85em] text-fg">{children}</code>
}
