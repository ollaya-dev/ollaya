/**
 * /design: a demo of the site's next design, built from the same data as the live pages. Not in
 * the sitemap, noindex. Dark only for now; the tokens are scoped to .rd so app.css is untouched.
 */
import type { Child } from 'hono/jsx'
import { Icon } from '../components/Icon'
import { LogoMark } from '../components/Logo'
import { CopyButton } from '../components/ui'
import { highlight } from '../lib/highlight'
import { catalog } from '../data/catalog'
import { docPages } from '../generated/content'
import { benchGpu, result } from '../data/numbers'
import stats from '../data/stats.json'
import { asset } from '../lib/assets'
import { benchUrl } from '../lib/bench'
import { githubStars } from '../lib/stars'
import { AUTHOR, GITHUB_URL, HF_URL, ISSUES_URL } from '../site'
import { hero, heroOf, type HeroQuestion } from '../data/hero'
import heroExamples from '../data/hero-examples.json'
import triage from '../../../crates/ollaya-api/src/presets/triage.json'
import { compatRequest, compatResponse, JEV, OLLAMA, speedup, vs, vsFeatures } from './home'
import { publicBench, shortName } from '../data/results'

const INSTALL = 'curl -fsSL https://ollaya.dev/install.sh | sh'
const lastWeek = stats.weeks.at(-1)!
const fmt = (n: number) => n.toLocaleString('en-US')
const mostAccurate = vs('pplx-decider:flash', 'ollaya')

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
      <svg viewBox="0 0 24 24" class="size-6 shrink-0" fill="currentColor" aria-hidden="true">
        <path d={p.path} />
      </svg>
    )
  return <img src={asset(`logos/${p.png}.png`)} class="size-6 shrink-0 rounded-[3px] brightness-150 grayscale" alt="" width="24" height="24" />
}

const tokens = `
.rd {
  color-scheme: dark;
  --color-canvas: #0a0a0a; --color-fg: #fafafa; --color-body: #c4c4c4; --color-muted: #8a8a8a; --color-faint: #5c5c5c;
  --color-line: rgb(255 255 255 / 0.1); --color-line-strong: rgb(255 255 255 / 0.2); --color-subtle: rgb(255 255 255 / 0.025);
  --color-code: rgb(255 255 255 / 0.04); --color-fill: rgb(255 255 255 / 0.06); --color-fill-strong: rgb(255 255 255 / 0.1);
  --color-btn: #fafafa; --color-btn-hover: #ffffff; --color-btn-fg: #0a0a0a; --color-term: rgb(10 10 10 / 0.78);
  --color-tok-comment: #6b6b6b; --color-tok-string: #a9c7a0; --color-tok-number: #e3d7b4; --color-tok-keyword: #c4c4c4;
  --color-tok-key: #fafafa; --color-tok-fn: #8cc8ff; --color-tok-punct: #6b6b6b;
  --color-us: #8cc8ff; --color-them: #f0a060; --color-pass: #3fb950; --color-fail: #f85149;
  background: var(--color-canvas); color: var(--color-body); overflow-x: clip;
}
/* The frame: two rails down the page, with a cross where a section's rule meets them. */
.rd .frame { position: relative; max-width: 72rem; margin: 0 auto; border-left: 1px solid var(--color-line); border-right: 1px solid var(--color-line); }
.rd .rule { position: relative; border-top: 1px solid var(--color-line); }
.rd .rule::before, .rd .rule::after { content: ''; position: absolute; top: -7px; width: 13px; height: 13px; pointer-events: none;
  background:
    linear-gradient(var(--color-line-strong), var(--color-line-strong)) 6px 0 / 1px 100% no-repeat,
    linear-gradient(var(--color-line-strong), var(--color-line-strong)) 0 6px / 100% 1px no-repeat; }
.rd .rule::before { left: -7px; } .rd .rule::after { left: calc(100% - 6px); }
/* A panel: a surface with a hairline, blurred over whatever colour sits behind it. */
.rd .panel { background: rgb(255 255 255 / 0.03); border: 1px solid var(--color-line); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); }
.rd .panel-strong { background: rgb(12 12 12 / 0.78); }
/* The grain: an SVG noise tile over everything, blended so it only breaks up the flat colour. */
.rd .grain { position: fixed; inset: 0; z-index: 50; pointer-events: none; opacity: 0.16; mix-blend-mode: screen;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='1.1' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size: 160px 160px; }
/* Blurred colour fields behind the hero and the closer: what the panels blur. */
.rd .field { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
.rd .ascii { position: absolute; inset: 0; pointer-events: none; mask-image: linear-gradient(180deg, #000 55%, transparent); -webkit-mask-image: linear-gradient(180deg, #000 55%, transparent); }
.rd .ascii canvas { display: block; }
@media (max-width: 640px) { .rd .ascii { opacity: 0.45; } }
.rd .field i { position: absolute; display: block; border-radius: 9999px; filter: blur(90px); opacity: 0.5; }
.rd .field .a { width: 60vw; height: 60vw; left: -10vw; top: -20vw; background: #2d5a4a; }
.rd .field .b { width: 50vw; height: 50vw; right: -12vw; top: -8vw; background: #5a4a2d; }
.rd .field .c { width: 45vw; height: 45vw; left: 30vw; top: 20vw; background: #2d3f5a; opacity: 0.4; }
@media (prefers-reduced-motion: no-preference) {
  .rd .field i { animation: drift 22s ease-in-out infinite alternate; }
  .rd .field .b { animation-duration: 28s; animation-delay: -9s; }
  .rd .field .c { animation-duration: 34s; animation-delay: -17s; }
}
@keyframes drift { to { transform: translate3d(6vw, 4vw, 0) scale(1.08); } }
.rd .term-glow { box-shadow: 0 1px 0 0 rgb(255 255 255 / 0.06) inset, 0 40px 120px -40px rgb(0 0 0 / 0.9); }
@media (prefers-reduced-motion: no-preference) {
  .rd .fill { transform-origin: left; animation: fill 0.7s cubic-bezier(.2,.7,.2,1) both; animation-delay: calc(0.6s + var(--row, 0) * 0.18s); }
}
@keyframes fill { from { transform: scaleX(0); } }
/* The publishers' marks on an endless belt: the track is laid twice and slides by half its width. */
.rd .marquee { overflow: hidden; mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent); -webkit-mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent); }
.rd .marquee-track { display: flex; width: max-content; gap: 3.5rem; padding-right: 3.5rem; }
@media (prefers-reduced-motion: no-preference) {
  .rd .marquee-track { animation: marquee 38s linear infinite; }
  .rd .marquee:hover .marquee-track { animation-play-state: paused; }
}
@keyframes marquee { to { transform: translateX(-50%); } }
/* The hero's example with every option showing: nothing folds, the extra options appear. */
.rd .is-open .term-fold { animation: term-in 0.4s ease-out both; animation-delay: calc(0.35s + var(--row, 0) * 0.05s); height: auto; opacity: 1; }
.rd .was-open:not(.is-open) .term-fold { animation: none; height: 0; opacity: 0; }
.rd .more { display: none; }
.rd .is-open .more { display: flex; }
/* Code with line numbers: one span per line, numbered by a counter in the gutter. */
.rd .code { counter-reset: l; font-size: 13px; line-height: 1.7; }
.rd .code .ln { display: grid; grid-template-columns: 2.5rem 1fr; }
.rd .code .ln::before { counter-increment: l; content: counter(l); color: var(--color-faint); text-align: right; padding-right: 1rem; user-select: none; }
.rd .code .tok-comment { font-style: normal; }
/* The scatter: Ollaya's models are filled dots, Ollama's are rings; the one under the pointer lights up. */
.rd .pt { cursor: default; outline: none; }
.rd .pt circle { transition: r 0.15s, opacity 0.15s; }
.rd .pt.is-on circle { r: 7; }
.rd .pt.is-on .dot { fill: var(--color-us); stroke: var(--color-us); }
.rd .pt.is-off { opacity: 0.12; pointer-events: none; }
.rd .pt:focus-visible circle { stroke: var(--color-us); stroke-width: 2; }
.rd .tip { position: absolute; z-index: 10; pointer-events: none; white-space: nowrap; }
/* The confidence meter: ten segments, lit in turn. */
.rd .meter { display: grid; grid-template-columns: repeat(10, 1fr); gap: 3px; }
.rd .meter i { display: block; height: 6px; border-radius: 1px; background: var(--color-fill-strong); }
.rd .meter i.on { background: var(--color-us); }
@media (prefers-reduced-motion: no-preference) {
  .rd .meter i.on { animation: lit 0.25s ease-out both; animation-delay: calc(0.6s + var(--row, 0) * 0.18s + var(--i, 0) * 0.04s); }
}
@keyframes lit { from { background: var(--color-fill-strong); } }
.rd ::selection { background: rgb(140 200 255 / 0.3); }
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
    <header class="fixed inset-x-0 top-0 z-40 border-b border-line bg-canvas/70 backdrop-blur-xl">
      <nav aria-label="Main" class="mx-auto flex h-14 max-w-6xl items-center gap-7 px-5">
        <a href="/" class="flex items-center gap-2 text-fg" aria-label="Ollaya home">
          <LogoMark class="size-7" />
          <span class="text-[15px] font-semibold tracking-tight">ollaya</span>
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
          <a href={GITHUB_URL} class="hidden h-9 items-center gap-1.5 rounded-md border border-line px-3 text-sm text-body hover:border-line-strong hover:text-fg sm:inline-flex">
            <Icon name="star" class="size-3.5" />
            {stars ?? 'GitHub'}
          </a>
          <a href="/download" class="inline-flex h-9 items-center rounded-md bg-btn px-4 text-sm font-medium text-btn-fg hover:bg-btn-hover">
            Download
          </a>
          <details class="group relative md:hidden">
            <summary class="inline-flex size-9 cursor-pointer items-center justify-center rounded-md border border-line text-fg" aria-label="Menu">
              <Icon name="menu" class="size-5 group-open:hidden" />
              <Icon name="close" class="hidden size-5 group-open:block" />
            </summary>
            <ul class="panel panel-strong absolute right-0 mt-2 w-48 rounded-md py-1 text-[15px]">
              {[...links, [GITHUB_URL, 'GitHub']].map(([href, label]) => (
                <li>
                  <a href={href} class="block px-4 py-2 text-body hover:text-fg">
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </details>
        </div>
      </nav>
    </header>
  )
}

function Install() {
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

function Hero() {
  return (
    <section class="relative overflow-hidden" aria-labelledby="rd-hero">
      <div class="field" aria-hidden="true">
        <i class="a"></i>
        <i class="b"></i>
        <i class="c"></i>
      </div>
      <div class="ascii" data-ascii aria-hidden="true"></div>
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
          <Install />
          <a href="/try" class="inline-flex h-11 items-center gap-1.5 rounded-md border border-line bg-canvas/40 px-4 text-sm font-medium text-fg hover:border-line-strong">
            Try it in the browser <Icon name="arrowRight" class="size-4" />
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
        <div class="border-b border-line p-5 md:border-r md:border-b-0">
          <p class="text-[13px] text-muted">A support message comes in</p>
          <blockquote class="term-row mt-3 text-[15px] leading-relaxed text-fg" style="--row:0">
            “{ex.message}”
          </blockquote>
          <p class="mt-5 text-[13px] text-muted">
            Ollaya asks <span class="font-mono text-[12px] text-body">{ex.model}</span> the {n} questions of the{' '}
            <span class="font-mono text-[12px] text-body">{ex.preset}</span> preset, in one request.
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
        <p class="px-6 pt-5 text-[13px] text-muted md:py-6">Models by</p>
        <div class="marquee py-5 md:py-6" aria-label={PUBLISHERS.map((p) => p.name).join(', ')}>
          <div class="marquee-track" aria-hidden="true">
            {[...PUBLISHERS, ...PUBLISHERS].map((p) => (
              <span class="inline-flex shrink-0 items-center gap-2.5 text-[15px] font-medium whitespace-nowrap text-body">
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

function DesignScatter({ W = 760, H = 420 }: { W?: number; H?: number }) {
  const dots = scatterDots()
  const m = { l: 44, r: 20, t: 16, b: 40 }
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
        {(['ollaya', 'ollama'] as const).map((srv) => (
          <button type="button" data-series={srv} aria-pressed="true" class="inline-flex items-center gap-2 rounded-md border border-line px-2.5 py-1 text-body hover:border-line-strong aria-[pressed=false]:opacity-40">
            <svg viewBox="0 0 12 12" class="size-3" aria-hidden="true">
              {srv === 'ollaya' ? <circle cx="6" cy="6" r="5" class="fill-fg" /> : <circle cx="6" cy="6" r="4" class="stroke-muted" fill="none" stroke-width="2" />}
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
        <text transform={`translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)`} text-anchor="middle" font-size="11" class="fill-muted">
          accuracy
        </text>
        {dots.map((d) => (
          <g
            class="pt"
            tabindex={0}
            data-name={d.name}
            data-server={d.server === 'ollaya' ? 'ollaya' : OLLAMA}
            data-acc={d.acc.toFixed(3)}
            data-ece={d.ece.toFixed(3)}
            data-ms={Math.round(d.ms)}
          >
            <title>{`${d.name} on ${d.server === 'ollaya' ? 'Ollaya' : OLLAMA}: accuracy ${d.acc.toFixed(3)}, ECE ${d.ece.toFixed(3)}, ${Math.round(d.ms)} ms`}</title>
            <circle cx={x(d.ms)} cy={y(d.acc)} r="12" fill="transparent" />
            {d.server === 'ollaya' ? (
              <circle cx={x(d.ms)} cy={y(d.acc)} r="5" class="dot fill-fg stroke-canvas" stroke-width="1.5" />
            ) : (
              <circle cx={x(d.ms)} cy={y(d.acc)} r="4.5" class="dot stroke-muted fill-canvas" stroke-width="2" />
            )}
            {d.label ? (
              <text x={x(d.ms) + 9} y={y(d.acc) + 4} font-size="12" class={d.server === 'ollaya' ? 'fill-fg' : 'fill-muted'}>
                {d.name}
              </text>
            ) : null}
          </g>
        ))}
      </svg>
      <div class="tip panel panel-strong rounded-md px-3 py-2 text-[13px] text-body [&_b]:font-medium [&_b]:text-fg [&_span]:text-muted" data-tip hidden></div>
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
    <a href={`/library/${m.name}`} class="group flex h-full flex-col bg-canvas p-6 transition-colors hover:bg-subtle">
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

function Closer() {
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
          <Install />
          <a href="/download" class="inline-flex h-11 items-center gap-2 rounded-md bg-btn px-5 text-sm font-medium text-btn-fg hover:bg-btn-hover">
            <Icon name="download" class="size-4" />
            Download
          </a>
        </div>
      </div>
    </section>
  )
}

/** The footer's link groups: every page the site has, by what the visitor is after. */
const docNav = (slug: string) => {
  const d = docPages.find((p) => p.slug === slug)
  if (!d) throw new Error(`design footer: no doc page ${slug}`)
  return { href: `/docs/${slug}`, label: d.nav }
}
const footerGroups: { title: string; links: { href: string; label: string }[] }[] = [
  {
    title: 'Product',
    links: [
      { href: '/download', label: 'Download' },
      { href: '/search', label: 'Models' },
      { href: '/try', label: 'Try in the browser' },
      { href: benchUrl(), label: 'Ollaya Bench' },
      { href: '/stats', label: 'Stats' },
      { href: '/blog', label: 'Blog' },
    ],
  },
  { title: 'Docs', links: ['quickstart', 'cli', 'api', 'modelfile', 'typesafe-compatibility', 'faq'].map(docNav) },
  { title: 'Use cases', links: ['agents', 'guardrails', 'routing', 'replace-llm-classifiers', 'cpu-classification', 'integrations'].map(docNav) },
  {
    title: 'Models',
    links: [...catalog]
      .sort((a, b) => a.rank - b.rank)
      .slice(0, 6)
      .map((m) => ({ href: `/library/${m.name}`, label: m.name })),
  },
  {
    title: 'Community',
    links: [
      { href: GITHUB_URL, label: 'GitHub' },
      { href: ISSUES_URL, label: 'Issues' },
      { href: HF_URL, label: 'Hugging Face' },
      { href: '/contributors', label: 'Contributors' },
      docNav('model-authors'),
      { href: '/llms.txt', label: 'llms.txt' },
    ],
  },
]

function Foot() {
  return (
    <footer class="frame">
      <div class="rule flex flex-col gap-6 px-6 py-10 md:flex-row md:items-start md:justify-between">
        <div class="max-w-sm">
          <a href="/" class="inline-flex items-center gap-2 text-fg" aria-label="Ollaya home">
            <LogoMark class="size-8" />
            <span class="text-lg font-semibold tracking-tight">ollaya</span>
          </a>
          <p class="mt-4 text-[15px] text-body">Run open decision models locally. Typed questions in, calibrated answers out, in milliseconds, on your own hardware.</p>
        </div>
        <div class="flex items-center gap-2">
          <a href={GITHUB_URL} class="inline-flex size-10 items-center justify-center rounded-md border border-line text-body hover:border-line-strong hover:text-fg" aria-label="GitHub">
            <svg viewBox="0 0 24 24" class="size-5" fill="currentColor" aria-hidden="true">
              <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
            </svg>
          </a>
          <a href={HF_URL} class="inline-flex h-10 items-center gap-2 rounded-md border border-line px-3 text-sm text-body hover:border-line-strong hover:text-fg">
            Hugging Face
          </a>
          <a href="/download" class="inline-flex h-10 items-center rounded-md bg-btn px-4 text-sm font-medium text-btn-fg hover:bg-btn-hover">
            Download
          </a>
        </div>
      </div>
      <nav aria-label="Footer" class="rule grid grid-cols-2 gap-px bg-line md:grid-cols-5">
        {footerGroups.map((g, i) => (
          <div class={`bg-canvas px-6 py-8 ${i === footerGroups.length - 1 ? 'col-span-2 md:col-span-1' : ''}`}>
            <h3 class="text-sm font-semibold text-fg">{g.title}</h3>
            <ul class="mt-4 space-y-2.5 text-[15px]" role="list">
              {g.links.map((l) => (
                <li>
                  <a href={l.href} class={`text-body hover:text-fg ${g.title === 'Models' ? 'font-mono text-[14px]' : ''}`}>
                    {l.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
      <div class="rule flex flex-col gap-3 px-6 py-6 text-[13px] text-muted sm:flex-row sm:items-center sm:justify-between">
        <p>
          © 2026 Ollaya · Apache-2.0 · Built by{' '}
          <a href={AUTHOR.url} class="text-body hover:text-fg">
            {AUTHOR.name}
          </a>
        </p>
        <p>An independent open-source project, not affiliated with Ollama or TypeSafe.</p>
      </div>
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
        <script src={asset('ascii.js')} defer></script>
        <script src={asset('chart.js')} defer></script>
        <script src={asset('demo.js')} defer></script>
      </head>
      <body class="rd min-h-screen font-sans antialiased" data-origin={origin}>
        <div class="grain" aria-hidden="true"></div>
        <Nav />
        <main class="border-b border-line">
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
