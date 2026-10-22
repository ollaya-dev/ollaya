// The /try demo (src/pages/try.tsx). The model and the asset URLs come from #try-app's data attributes, the
// examples from #try-examples. The runtime (web/ollaya.js with onnxruntime-web's WebAssembly build) is a separate
// bundle, /static/ollaya-wasm.js, imported only when the visitor presses Load. No WebGPU: the 2-bit graph's
// MatMulNBits ran slower there than on WebAssembly (1.8 s against 0.24 s a call).

const app = document.getElementById('try-app')
const examples = JSON.parse(document.getElementById('try-examples').textContent)
const $ = (sel) => app.querySelector(sel)
const d = app.dataset

// ?repo=<org/name>&endpoint=<url> override the model, for testing.
const params = new URLSearchParams(location.search)
const repo = params.get('repo') || d.repo
const endpoint = params.get('endpoint') || 'https://huggingface.co'
if (params.get('repo')) {
  $('[data-model-name]').textContent = repo
  $('[data-model-repo]').textContent = `${new URL(endpoint).host}/${repo}`
  $('[data-model-repo]').href = `${endpoint}/${repo}`
}

// Three questions over a state of about 200 tokens, the size of an example.
const WARM_UP = {
  state: 'The order arrived late and the box was damaged. '.repeat(20),
  questions: Object.fromEntries(
    ['a', 'b', 'c'].map((id) => [id, { type: 'choice', instructions: 'Which fits best?', criteria: { x: 'one', y: 'two', z: 'three' } }]),
  ),
}

const status = (text) => ($('[data-status]').textContent = text)
const mb = (n) => (n / 1e6).toFixed(1)
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`)

let model
let request = { state: examples[0].state, questions: examples[0].questions }

// --- The request -----------------------------------------------------------------------------------------

const TYPE_LABEL = { noul: 'yes / no', choice: 'choice', score: 'score' }
const pill =
  'inline-flex items-center rounded-full border border-line-strong px-2 py-[1px] text-[11px] font-medium text-muted'

// Same markup as Questions in try.tsx.
function drawQuestions() {
  $('[data-questions]').innerHTML = `<ul class="divide-y divide-line" role="list">${Object.entries(request.questions)
    .map(([id, q]) => {
      const options = Array.isArray(q.criteria) ? q.criteria : Object.keys(q.criteria ?? {})
      return `<li class="py-3 first:pt-0 last:pb-0">
        <p class="flex flex-wrap items-baseline gap-x-2 gap-y-1"><span class="font-mono text-[13px] text-fg">${esc(id)}</span>
        <span class="${pill}">${esc(TYPE_LABEL[q.type] ?? q.type)}</span></p>
        ${q.instructions ? `<p class="mt-1 text-sm text-body">${esc(q.instructions)}</p>` : ''}
        ${q.type !== 'noul' && options.length ? `<p class="mt-1.5 text-[13px] text-muted">${options.map((o) => esc(typeof o === 'string' ? o : JSON.stringify(o))).join(' · ')}</p>` : ''}
      </li>`
    })
    .join('')}</ul>`
}

// Same markup as Pending in try.tsx.
function drawPending() {
  $('[data-latency]').textContent = ''
  $('[data-answers]').innerHTML = `<ul class="divide-y divide-line" role="list">${Object.keys(request.questions)
    .map(
      (id) => `<li class="py-4 first:pt-0 last:pb-0">
        <p class="flex items-baseline justify-between gap-3"><span class="font-mono text-[13px] text-fg">${esc(id)}</span>
        <span class="text-sm text-muted tabular-nums">–</span></p>
        <span class="mt-2 block h-1.5 rounded-full bg-fill-strong" aria-hidden="true"></span></li>`,
    )
    .join('')}</ul>`
}

const writeJson = () => ($('[data-json]').value = JSON.stringify(request, null, 2))

app.querySelectorAll('[data-example]').forEach((btn) =>
  btn.addEventListener('click', () => {
    const ex = examples.find((e) => e.key === btn.dataset.example)
    app.querySelectorAll('[data-example]').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)))
    request = { state: ex.state, questions: ex.questions }
    $('[data-state]').value = ex.state
    $('[data-json-error]').textContent = ''
    writeJson()
    drawQuestions()
    drawPending()
    if (model) decide()
  }),
)

// A shared link carries its request as JSON in the hash (Copy link). It is only drawn, escaped, until the visitor
// loads the model; the link never carries ?repo=, so it always opens on the page's own model.
function shared() {
  try {
    const r = JSON.parse(decodeURIComponent(location.hash.slice(1)))
    if (typeof r?.state === 'string' && r.questions && typeof r.questions === 'object') return r
  } catch {}
}
const fromLink = location.hash.length > 1 && shared()
if (fromLink) {
  request = { state: fromLink.state, questions: fromLink.questions }
  app.querySelectorAll('[data-example]').forEach((b) => b.setAttribute('aria-pressed', 'false'))
  $('[data-state]').value = request.state
  writeJson()
  drawQuestions()
  drawPending()
}

$('[data-share]').addEventListener('click', async () => {
  const url = `${location.origin}${location.pathname}#${encodeURIComponent(JSON.stringify(request))}`
  history.replaceState(null, '', url)
  const ok = await navigator.clipboard.writeText(url).then(
    () => true,
    () => false,
  )
  const label = $('[data-share-label]')
  label.textContent = ok ? 'Copied' : 'Link in the address bar'
  setTimeout(() => (label.textContent = 'Copy link'), 2000)
})

$('[data-state]').addEventListener('input', (e) => {
  request = { ...request, state: e.target.value }
  writeJson()
  schedule()
})

$('[data-json]').addEventListener('input', (e) => {
  let next
  try {
    next = JSON.parse(e.target.value)
    if (!next || typeof next !== 'object' || !next.questions || typeof next.questions !== 'object') {
      throw new Error('the request needs a "questions" object')
    }
  } catch (err) {
    $('[data-json-error]').textContent = `Not used yet: ${err.message}`
    return
  }
  $('[data-json-error]').textContent = ''
  request = next
  if (typeof next.state === 'string') $('[data-state]').value = next.state
  drawQuestions()
  schedule()
})

// --- The answers -----------------------------------------------------------------------------------------

const pct = (p) => `${(p * 100).toFixed(1)}%`

function bar(p, on) {
  return `<span class="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-fill-strong" aria-hidden="true">
    <span class="block h-full rounded-full ${on ? 'bg-us' : 'bg-faint'}" style="width:${pct(p)}"></span></span>`
}

// One option row: label, bar, probability; the answer is bright, the others dim (as on the home page).
function row(label, p, on) {
  return `<li class="flex items-center gap-3 text-[13px]">
    <span class="w-28 shrink-0 truncate sm:w-44 ${on ? 'font-medium text-fg' : 'text-muted'}" title="${esc(label)}">${esc(label)}${on ? '<span class="sr-only">, the answer</span>' : ''}</span>
    ${bar(p, on)}
    <span class="w-[4.5ch] text-right tabular-nums ${on ? 'text-fg' : 'text-muted'}">${p.toFixed(2)}</span></li>`
}

function answerHtml(id, a, q) {
  let head
  let body = ''
  if (a.type === 'noul') {
    head = `<span class="text-sm font-medium text-fg tabular-nums">${a.noul >= 0.5 ? 'Yes' : 'No'} · ${pct(a.noul)}</span>`
    body = `<div class="mt-2 flex">${bar(a.noul, true)}</div>`
  } else if (a.type === 'choice') {
    head = `<span class="text-sm font-medium text-fg">${esc(a.choice)}</span>`
    body = `<ol class="mt-2 space-y-1.5">${Object.entries(a.probabilities)
      .map(([k, p]) => row(k, p, k === a.choice))
      .join('')}</ol>`
  } else if (a.type === 'score') {
    const levels = Object.keys(a.probabilities).length
    const top = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])[0][0]
    head = `<span class="text-sm font-medium text-fg tabular-nums">${a.score.toFixed(2)} <span class="font-normal text-muted">of ${levels - 1}</span></span>`
    body = `<ol class="mt-2 space-y-1.5">${Object.entries(a.probabilities)
      .map(([k, p]) => row(`${k} · ${a.legend?.[k] ?? ''}`, p, k === top))
      .join('')}</ol>`
  } else {
    head = `<span class="text-sm text-muted">${esc(a.type)}</span>`
  }
  const instructions = q?.instructions ? `<p class="mt-0.5 text-[13px] text-muted">${esc(q.instructions)}</p>` : ''
  return `<li class="py-4 first:pt-0 last:pb-0">
    <p class="flex items-baseline justify-between gap-3"><span class="font-mono text-[13px] text-fg">${esc(id)}</span>${head}</p>
    ${instructions}${body}</li>`
}

let running = false
let again = false

async function decide() {
  if (running) return void (again = true)
  running = true
  const sent = request
  try {
    const t = performance.now()
    const r = await model.decide(sent)
    const ms = performance.now() - t
    $('[data-answers]').innerHTML = `<ul class="divide-y divide-line" role="list">${Object.entries(r.answers)
      .map(([id, a]) => answerHtml(id, a, sent.questions[id]))
      .join('')}</ul>`
    $('[data-latency]').textContent = `${Math.round(ms)} ms · ${r.usage?.input_tokens ?? '?'} tokens`
  } catch (e) {
    $('[data-latency]').textContent = ''
    $('[data-answers]').innerHTML = `<p class="text-sm text-fail">${esc(e.message ?? e)}</p>`
  } finally {
    running = false
    if (again) {
      again = false
      decide()
    }
  }
}

// Once loaded, edits re-run after a pause in typing.
let timer
function schedule() {
  if (!model) return
  clearTimeout(timer)
  timer = setTimeout(decide, 400)
}

$('[data-decide]').addEventListener('click', decide)

// --- Loading ---------------------------------------------------------------------------------------------

$('[data-load]').addEventListener('click', async () => {
  const btn = $('[data-load]')
  btn.disabled = true
  const progress = $('[data-progress]')
  const fill = $('[data-progress-bar]')
  const t = performance.now()
  try {
    status('Starting the runtime…')
    const { ort, fromHub, load } = await import(d.runtimeWasm)
    // Served from this site, so no third-party CDN is involved.
    ort.env.wasm.wasmPaths = { wasm: d.ortWasm, mjs: d.ortMjs }
    model = await load(fromHub(repo, { endpoint }), {
      name: d.name,
      wasm: d.ollayaWasm,
      onProgress: ({ loaded, total }) => {
        const p = total ? Math.min(1, loaded / total) : 0
        fill.style.width = `${(p * 100).toFixed(1)}%`
        progress.setAttribute('aria-valuenow', String(Math.round(p * 100)))
        status(`Downloading from Hugging Face: ${mb(loaded)} of ${mb(total)} MB`)
      },
    })
    fill.style.width = '100%'
    progress.setAttribute('aria-valuenow', '100')
    // Chrome runs onnxruntime's WebAssembly as baseline code until its kernels have run a while (the first calls
    // take seconds, then 0.2-0.3 s on a laptop): warm up until the calls stop getting faster, for at most 8 s.
    status('Warming up…')
    for (let prev = Infinity, end = performance.now() + 8000; performance.now() < end; ) {
      const s = performance.now()
      await model.decide(WARM_UP)
      const ms = performance.now() - s
      if (ms > 0.8 * prev) break
      prev = ms
    }
    const threads = ort.env.wasm.numThreads
    const backend = `WebAssembly${threads > 1 ? `, ${threads} threads` : ''}`
    status(`Loaded in ${((performance.now() - t) / 1000).toFixed(1)} s · ${backend}`)
    btn.hidden = true
    $('[data-decide]').disabled = false
    $('[data-decide-hint]').textContent = 'Edits re-run on their own.'
    decide()
  } catch (e) {
    status(`Could not load the model: ${e.message ?? e}`)
    btn.disabled = false
  }
})
