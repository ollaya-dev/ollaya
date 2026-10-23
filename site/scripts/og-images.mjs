// Renders the social preview cards (1200×630 PNG) into public/static/:
//   og.png           every page without its own card
//   og/<model>.png   a model's pages (/library/<model>, its tags)
// and into bench-public/static/:
//   og-bench.png     bench pages without a card of their own (og-bench.mjs makes those)
//
// The PNGs are committed, so the site build does not need a browser. Run this by hand after
// changing a card or adding a model:
//
//   node scripts/og-images.mjs [og/results.png ...]
//
// It needs a Chromium: $CHROME, else Playwright's cached headless shell. Geist is downloaded from
// Google Fonts, so it also needs the network (without it the cards fall back to system fonts).

import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { esc, inlineFonts, page as pageWith, renderCards, top } from './og-lib.mjs'

const root = fileURLToPath(new URL('..', import.meta.url))
const out = join(root, 'public', 'static')

/** Model cards. Keep the claims in step with content/library/<model>.md. */
const MODELS = [
  {
    name: 'laya',
    badge: 'Fastest',
    text: 'Typed, calibrated answers in 8–10 ms on a GPU, in English and 100+ languages.',
    by: 'Convai Innovations',
    license: 'Apache-2.0',
  },
  {
    name: 'decider',
    badge: 'Most accurate',
    text: 'The most accurate open decision model on typed decisions. Qwen3.5 decoders, 2B and 0.8B.',
    by: 'Mapika',
    license: 'Apache-2.0',
  },
  {
    name: 'nli',
    badge: 'Zero-shot',
    text: 'Zero-shot NLI classifiers on DeBERTa-v3-large and ModernBERT-large. The most accurate encoder.',
    by: 'Moritz Laurer',
    license: 'MIT · Apache-2.0',
  },
  {
    name: 'gliclass',
    badge: 'Zero-shot',
    text: 'An instruction-following zero-shot classifier that scores every option in one pass.',
    by: 'Knowledgator',
    license: 'Apache-2.0',
  },
]

const fonts = await inlineFonts()
const page = (body) => pageWith(fonts, body)

const command = (model) => `<div class="cmd"><span class="p">❯</span> <span class="c">ollaya</span> run ${esc(model)}</div>`

/** The results card's numbers, from the newest public-benchmark run in ../results (as on /results). */
function resultsCard() {
  const dir = join(root, '..', 'results', 'runs')
  // The newest run with Ollama on the same GPU (bench editions measure Ollaya alone).
  const runs = readdirSync(dir).filter((f) => f.includes('public-benchmark')).sort().map((f) => JSON.parse(readFileSync(join(dir, f), 'utf8')))
  const run = runs.filter((r) => r.servers?.ollama).at(-1)
  // Ollama's run holds Ollama's rows alone; Ollaya's are the newest row of each model across its runs on the same machine.
  const ours = new Map()
  for (const r of runs) if (!r.servers?.ollama && r.machine === run.machine) for (const x of r.results) ours.set(x.model, { server: 'ollaya', ...x })
  const results = [...run.results, ...ours.values()]
  const by = (server) => results.filter((r) => r.server === server).sort((a, b) => b.macro_accuracy - a.macro_accuracy)[0]
  const best = by('ollaya')
  const ollama = by('ollama')
  const nimble = results.find((r) => r.server === 'ollaya' && r.model.startsWith('nimble'))
  const nimbleOllama = results.find((r) => r.server === 'ollama' && r.model === 'nimble')
  const ms = (r) => `${Math.round(r.median_ms)} ms`
  return page(`${top('ollaya.dev/results')}
      <div class="main"><span class="badge">Results · RTX 5090 · 3,880 questions</span><h1 style="margin-top:22px">Measured, not claimed.</h1>
      <p class="text"><b style="color:#fafafa;font-weight:500">${esc(best.model)}</b>: ${best.macro_accuracy.toFixed(3)} accuracy at ${ms(best)}.
      Ollama's best: ${ollama.macro_accuracy.toFixed(3)} at ${ms(ollama)}. Same Nimble weights: calibration error
      ${nimble.pooled_ece.toFixed(3)} on Ollaya, ${nimbleOllama.pooled_ece.toFixed(3)} on Ollama.</p></div>
      <div class="bottom"><div class="meta" style="text-align:left"><b>Accuracy · calibration · speed · parity</b><br>every model, on our own GPUs and CPUs, with the raw data</div></div>`)
}

const cards = [
  {
    file: 'og-bench.png',
    dir: join(root, 'bench-public', 'static'),
    html: page(`${top('bench.ollaya.dev', 'ollaya <span style="color:#8a8a8a">bench</span>')}
      <div class="main"><span class="badge">Accuracy · calibration · speed · memory</span>
      <h1 style="margin-top:22px">Every open decision model, measured on real hardware.</h1>
      <p class="text">One released runtime, the authors' own weights, the same 5,880 questions, and GPUs and CPUs people own. Every number with its raw data.</p></div>`),
  },
  { file: 'og/results.png', html: resultsCard() },
  {
    file: 'og.png',
    html: page(`${top('ollaya.dev')}
      <div class="main"><h1>Run decision models locally.</h1>
      <p class="text">Typed questions in, calibrated answers out, in milliseconds. Private and open source.</p></div>
      <div class="bottom">${command('laya')}<div class="meta"><b>laya · decider · nli · gliclass</b><br>Apache-2.0</div></div>`),
  },
  ...MODELS.map((m) => ({
    file: `og/${m.name}.png`,
    html: page(`${top(`ollaya.dev/library/${m.name}`)}
      <div class="main"><span class="badge">${esc(m.badge)}</span><h1 class="model">${esc(m.name)}</h1>
      <p class="text">${esc(m.text)}</p></div>
      <div class="bottom">${command(m.name)}<div class="meta">by <b>${esc(m.by)}</b><br>${esc(m.license)}</div></div>`),
  })),
]

// `node scripts/og-images.mjs og/results.png` renders only the cards named.
const only = process.argv.slice(2)
await renderCards(cards.filter((c) => !only.length || only.includes(c.file)), out)
