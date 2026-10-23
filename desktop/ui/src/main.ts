// Ollaya desktop: an overview of the local server (its address, endpoints, speed and the latest
// requests) and the model library as cards, each opening a model page with downloads, a panel to
// try it and its model card. Plain DOM, rendered from one state object.
import {
  backend,
  type ActivityResponse,
  type Answer,
  type DecideResponse,
  type LibraryModel,
  type Preset,
  type Status,
} from './backend'

// --- tiny DOM helper ------------------------------------------------------------------------------

type Child = Node | string | null | undefined | false
type Attrs = Record<string, string | boolean | ((e: Event) => void) | undefined>

function h(tag: string, attrs: Attrs = {}, ...children: Child[]): HTMLElement {
  const el = document.createElement(tag)
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === false) continue
    if (typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v)
    else if (v === true) el.setAttribute(k, '')
    else el.setAttribute(k, v)
  }
  for (const c of children) if (c) el.append(c)
  return el
}

/** Ollaya's mark, the gaze: two rings and one pupil (site/src/components/Logo.tsx). */
const GAZE = `<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="9.5" cy="16" r="6.5" fill="none" stroke="currentColor" stroke-width="3.2"/><circle cx="22.5" cy="16" r="6.5" fill="none" stroke="currentColor" stroke-width="3.2"/><circle cx="22.5" cy="16" r="2.6" fill="currentColor"/></svg>`
const svg = (body: string, width = 1.6) =>
  `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`
const ICON = {
  download: svg('<path d="M10 3v10m0 0 4-4m-4 4-4-4M4 16h12"/>'),
  check: svg('<path d="m5 10.5 3.2 3.2L15 7"/>', 1.8),
  x: svg('<path d="m6 6 8 8M14 6l-8 8"/>'),
  back: svg('<path d="M16 10H4m0 0 5-5m-5 5 5 5"/>'),
  external: svg('<path d="M11 4h5v5m0-5-7 7M8 5H5a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-3"/>'),
  copy: svg('<rect x="7" y="7" width="9" height="9" rx="1.5"/><path d="M13 7V5.5A1.5 1.5 0 0 0 11.5 4h-6A1.5 1.5 0 0 0 4 5.5v6A1.5 1.5 0 0 0 5.5 13H7"/>'),
  overview: svg('<path d="M3 10h3l2-5 4 10 2-5h3"/>'),
  models: svg('<rect x="3.5" y="3.5" width="5" height="5" rx="1"/><rect x="11.5" y="3.5" width="5" height="5" rx="1"/><rect x="3.5" y="11.5" width="5" height="5" rx="1"/><rect x="11.5" y="11.5" width="5" height="5" rx="1"/>'),
  search: svg('<circle cx="9" cy="9" r="5"/><path d="m16 16-3.5-3.5"/>'),
  spinner: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true" class="spin"><circle cx="10" cy="10" r="7" stroke="currentColor" stroke-opacity=".25" stroke-width="2"/><path d="M17 10a7 7 0 0 0-7-7" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
}
function icon(markup: string, cls: string): HTMLElement {
  const s = h('span', { class: `inline-flex shrink-0 ${cls}` })
  s.innerHTML = markup // constant markup, never data
  return s
}

// --- state ----------------------------------------------------------------------------------------

interface Pulling {
  completed: number
  total: number
  status: string
}

type View = { page: 'overview' } | { page: 'models' } | { page: 'model'; name: string }

const state = {
  view: { page: 'overview' } as View,
  status: null as Status | null,
  statusBusy: false,
  statusError: '',
  /** A newer Ollaya release than this app. */
  update: null as string | null,
  library: [] as LibraryModel[],
  libraryError: '',
  installed: new Set<string>(),
  loaded: [] as string[],
  activity: null as ActivityResponse | null,
  activityError: '',
  query: '',
  onlyInstalled: false,
  pulling: new Map<string, Pulling>(),
  pullError: '',
  presets: [] as Preset[],
  /** Per installed version: the ids of its built-in questions, or null when it takes questions. */
  builtin: new Map<string, string[] | null>(),
  runModel: '',
  preset: 'triage',
  input: '',
  customQuestions: '',
  running: false,
  result: null as DecideResponse | null,
  runError: '',
  copied: '',
}

/** What `/api/decide` with `preset: triage` is shown with when someone asks for an example. */
const EXAMPLE = "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."

/**
 * The versions to offer: `<model>:latest` first (what `ollaya run <model>` uses), then the main tags.
 * Precision variants (-fp16, -fp32) are for pinning, not for picking.
 */
const mainTags = (m: LibraryModel) => {
  const tags = m.tags.map((t) => t.name).filter((n) => !/-fp(16|32)$/.test(n))
  const latest = `${m.name}:latest`
  return [latest, ...tags.filter((t) => t !== latest)]
}
const tagOf = (m: LibraryModel, tag: string) => m.tags.find((t) => t.name === tag)
const installedTags = (m: LibraryModel) => mainTags(m).filter((t) => state.installed.has(t))
const modelNamed = (name: string) => state.library.find((m) => m.name === name)
const title = (m: LibraryModel) => m.title || m.name

/** `builtinOf` requests, one per version. */
const asking = new Map<string, Promise<string[] | null>>()

/**
 * The ids of the questions `model` asks by itself (qwen3guard: safety, category, …), or null when
 * every request brings questions. Such a model rejects a preset or custom questions.
 */
function builtinOf(model: string): Promise<string[] | null> {
  let p = asking.get(model)
  if (!p) {
    p = backend
      .builtinQuestions(model)
      .then(
        (q) => (q ? Object.keys(q) : null),
        () => (asking.delete(model), null), // unknown for now: ask again on the next run
      )
      .then((ids) => {
        state.builtin.set(model, ids)
        render()
        return ids
      })
    asking.set(model, p)
  }
  return p
}

/** Forget what a version asks, once it is downloaded again or removed. */
function forget(tag: string) {
  asking.delete(tag)
  state.builtin.delete(tag)
}

// --- actions --------------------------------------------------------------------------------------

/** Poll the server; redraw only when what is shown changed, so a caret or a selection survives. */
async function refresh() {
  const before = JSON.stringify([state.status, [...state.installed], state.loaded, state.activity, state.activityError])
  state.status = await backend.status()
  if (state.status.running) {
    const [installed, loaded, activity] = await Promise.allSettled([backend.installed(), backend.loaded(), backend.activity()])
    state.installed = new Set(installed.status === 'fulfilled' ? installed.value.map((m) => m.name) : [])
    state.loaded = loaded.status === 'fulfilled' ? loaded.value : []
    state.activity = activity.status === 'fulfilled' ? activity.value : null
    // A server older than 0.17.0 (a command-line install) has no activity log.
    state.activityError = activity.status === 'rejected' ? String(activity.reason) : ''
  } else {
    state.installed = new Set()
    state.loaded = []
    state.activity = null
  }
  // The overview has no field to keep, and its per-second rate moves with the clock.
  if (state.view.page === 'overview' || JSON.stringify([state.status, [...state.installed], state.loaded, state.activity, state.activityError]) !== before) render()
}

async function toggleServer() {
  state.statusBusy = true
  state.statusError = ''
  render()
  try {
    state.status = state.status?.running ? await backend.stopServer() : await backend.startServer()
  } catch (e) {
    state.statusError = String(e)
  }
  state.statusBusy = false
  await refresh()
  render()
}

async function pull(tag: string) {
  state.pullError = ''
  state.pulling.set(tag, { completed: 0, total: 0, status: 'starting' })
  render()
  try {
    await backend.pull(tag)
  } catch (e) {
    state.pullError = `${tag}: ${e}`
  }
  state.pulling.delete(tag)
  forget(tag)
  await refresh()
  render()
}

async function remove(tag: string) {
  try {
    await backend.remove(tag)
  } catch (e) {
    state.pullError = String(e)
  }
  forget(tag)
  await refresh()
  if (state.runModel === tag) state.runModel = ''
  render()
}

async function run() {
  const model = state.runModel
  if (!model || !state.input.trim()) return
  state.running = true
  state.runError = ''
  render()
  try {
    // A model with built-in questions gets the state alone.
    state.result = (await builtinOf(model))
      ? await backend.decide(model, state.input, null, null)
      : state.preset === 'custom'
        ? await backend.decide(model, state.input, null, state.customQuestions)
        : await backend.decide(model, state.input, state.preset, null)
  } catch (e) {
    state.result = null
    state.runError = String(e)
  }
  state.running = false
  render()
  // Bring the answers into view inside the main pane (they land below the fold on a small window).
  const pane = document.getElementById('main-pane')
  const answers = document.getElementById('answers')
  if (pane && answers) {
    const below = answers.getBoundingClientRect().bottom + 24 - pane.getBoundingClientRect().bottom
    if (below > 0) pane.scrollTop += below
  }
}

function go(view: View) {
  document.getElementById('main-pane')?.scrollTo({ top: 0 })
  if (view.page === 'model' && !(state.view.page === 'model' && state.view.name === view.name)) {
    state.runModel = ''
    state.result = null
    state.runError = ''
  }
  state.view = view
  render()
}

async function copy(text: string) {
  try {
    await navigator.clipboard.writeText(text)
  } catch {
    // Older webviews (WebKitGTK) without the async clipboard.
    const area = h('textarea', {}) as HTMLTextAreaElement
    area.value = text
    document.body.append(area)
    area.select()
    document.execCommand('copy')
    area.remove()
  }
  state.copied = text
  render()
  setTimeout(() => {
    if (state.copied === text) (state.copied = ''), render()
  }, 1500)
}

// --- formatting -----------------------------------------------------------------------------------

/** Bytes as `ollaya list` prints them. */
function bytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`
  return `${Math.max(1, Math.round(n / 1e3))} KB`
}

function ms(n: number): string {
  return n < 10 ? n.toFixed(1) : String(Math.round(n))
}

function percent(p: { completed: number; total: number }): string {
  return p.total > 0 ? `${Math.floor((p.completed / p.total) * 100)}%` : '…'
}

/** What the activity log says about speed: the median answer time, and requests per second. */
function speed(a: ActivityResponse) {
  const times = a.requests
    .filter((r) => !r.error)
    .map((r) => r.total_duration / 1e6)
    .sort((x, y) => x - y)
  const median = times.length ? times[Math.floor(times.length / 2)]! : null
  // The rate between the first and the last request of the past minute: the log holds 100
  // requests, which a busy client sends in a few seconds.
  const at = a.requests.map((r) => Date.parse(r.at)).filter((t) => Date.now() - t < 60_000)
  const seconds = at.length > 1 ? Math.max(1, (at[0]! - at.at(-1)!) / 1000) : 60
  return { median, perSecond: at.length > 1 ? (at.length - 1) / seconds : at.length / 60, samples: times.length }
}

// --- view: frame ----------------------------------------------------------------------------------

const btnPrimary =
  'inline-flex h-8 items-center justify-center gap-1.5 rounded-md bg-btn px-3.5 text-[13px] font-medium text-btn-fg hover:bg-btn-hover disabled:opacity-50'
const btnSecondary =
  'inline-flex h-8 items-center justify-center gap-1.5 rounded-md border border-line px-3.5 text-[13px] font-medium text-fg hover:border-line-strong hover:bg-fill disabled:opacity-50'
const btnSmall =
  'inline-flex h-7 items-center gap-1 rounded-md border border-line px-2.5 text-xs font-medium text-fg hover:border-line-strong hover:bg-fill disabled:opacity-50'
const field =
  'rounded-md border border-line bg-canvas/60 text-fg placeholder:text-muted hover:border-line-strong focus:border-line-strong focus:ring-2 focus:ring-fill-strong focus:outline-none'
const card = 'rounded-md border border-line'
const h2 = 'text-sm font-semibold text-fg'

/** What the server's device means, for the status line's tooltip (#44). */
function deviceHint(device: string | null): string {
  if (device === 'CPU only')
    return "The app's built-in server runs every model on the CPU. Install the command-line Ollaya with its GPU pack (install.ps1 or install.sh) and the app starts the server from there, on your NVIDIA GPU."
  if (device === 'GPU') return 'The server runs from the command-line install, which has the GPU pack.'
  return ''
}

function header(): HTMLElement {
  const s = state.status
  const running = !!s?.running
  return h(
    'header',
    { class: 'flex h-14 shrink-0 items-center justify-between border-b border-line bg-canvas/70 px-5 backdrop-blur-xl' },
    h(
      'div',
      { class: 'flex items-center gap-2 text-fg' },
      icon(GAZE, 'size-7'),
      h('span', { class: 'text-[17px] font-semibold tracking-[-0.035em]' }, 'ollaya'),
    ),
    h(
      'div',
      { class: 'flex items-center gap-3' },
      state.statusError ? h('span', { class: 'max-w-80 truncate text-xs text-bad', title: state.statusError }, state.statusError) : null,
      state.update
        ? h(
            'button',
            { class: btnSmall, title: 'Opens the download page', onClick: () => void backend.openDownload() },
            `Ollaya ${state.update} is available`,
          )
        : null,
      h(
        'span',
        { class: 'flex items-center gap-2 text-[13px] text-muted', title: running ? deviceHint(s?.device ?? null) : '' },
        h('span', { class: `size-2 rounded-full ${running ? 'bg-ok' : 'bg-faint'}` }),
        s === null ? 'Checking…' : running ? ['Running', s.device].filter(Boolean).join(' · ') : 'Stopped',
      ),
      h(
        'button',
        { class: running ? btnSecondary : btnPrimary, disabled: state.statusBusy || s === null, onClick: toggleServer },
        state.statusBusy ? icon(ICON.spinner, 'size-3.5') : null,
        running ? 'Stop' : 'Start',
      ),
    ),
  )
}

function sidebar(): HTMLElement {
  const item = (label: string, svgIcon: string, view: View, on: boolean, note?: string) =>
    h(
      'button',
      {
        class: `flex h-9 w-full items-center gap-2.5 rounded-md px-2.5 text-left text-[13px] ${on ? 'bg-fill font-medium text-fg' : 'text-body hover:bg-fill hover:text-fg'}`,
        'aria-current': on ? 'page' : undefined,
        onClick: () => go(view),
      },
      icon(svgIcon, 'size-4'),
      h('span', { class: 'flex-1' }, label),
      note ? h('span', { class: 'text-xs text-muted tabular-nums' }, note) : null,
    )
  const page = state.view.page
  const have = state.library.filter((m) => installedTags(m).length).length
  return h(
    'nav',
    { class: 'flex w-52 shrink-0 flex-col justify-between border-r border-line p-2.5', 'aria-label': 'Pages' },
    h(
      'div',
      { class: 'space-y-0.5' },
      item('Overview', ICON.overview, { page: 'overview' }, page === 'overview'),
      item('Models', ICON.models, { page: 'models' }, page !== 'overview', have ? String(have) : undefined),
    ),
    h('p', { class: 'px-2.5 pb-1 text-[11px] text-muted' }, state.status?.version ? `Ollaya ${state.status.version}` : ''),
  )
}

/** A page is the website's frame: two rails, and bands one under the other, a rule between them. */
function main(): HTMLElement {
  const v = state.view
  const bands = v.page === 'overview' ? overview() : v.page === 'models' ? models() : modelPage(modelNamed(v.name))
  return h(
    'main',
    { id: 'main-pane', class: 'relative min-w-0 flex-1 overflow-y-auto px-6' },
    h('div', { class: 'frame mx-auto flex min-h-full w-full max-w-4xl flex-col pb-10' }, ...bands),
  )
}

/** A page's opening band: its title, a lead, and whatever goes under them. */
function head(title: Child, lead: Child, ...rest: Child[]): HTMLElement {
  return h(
    'header',
    { class: 'px-8 pt-10 pb-8' },
    h('h1', { class: 'text-3xl font-semibold tracking-[-0.03em] text-fg' }, title),
    lead ? h('p', { class: 'mt-2 max-w-2xl text-[15px] text-body' }, lead) : null,
    ...rest,
  )
}

/** A band under the head, with a rule above it; `title` names it for screen readers too. */
function band(title: string | null, ...children: Child[]): HTMLElement {
  const id = title ? `band-${title.toLowerCase().replace(/\W+/g, '-')}` : undefined
  return h('section', { class: 'rule px-8 py-7', 'aria-labelledby': id }, title ? h('h2', { id, class: h2 }, title) : null, ...children)
}

function copyButton(text: string, label: string): HTMLElement {
  const done = state.copied === text
  return h(
    'button',
    { class: 'inline-flex size-7 shrink-0 items-center justify-center rounded-md text-muted hover:bg-fill hover:text-fg', title: done ? 'Copied' : label, 'aria-label': label, onClick: () => void copy(text) },
    icon(done ? ICON.check : ICON.copy, `size-4 ${done ? 'text-ok' : ''}`),
  )
}

// --- view: overview -------------------------------------------------------------------------------

function overview(): HTMLElement[] {
  const s = state.status
  const top = head('Overview', 'The server on this computer, and what it has answered.')
  if (!s?.running)
    return [
      top,
      band(
        null,
        h(
          'div',
          { class: 'flex items-center justify-between gap-4' },
          h('p', { class: 'text-sm text-muted' }, s === null ? 'Checking the server…' : 'The server is stopped. Start it to download and run models.'),
          h('button', { class: btnPrimary, disabled: state.statusBusy || s === null, onClick: toggleServer }, 'Start'),
        ),
      ),
    ]
  return [top, stats(), server(s.url), endpoints(s.url), recent()]
}

/** The numbers as the website's proof strip: cells sharing their borders. */
function stats(): HTMLElement {
  const a = state.activity
  const sp = a ? speed(a) : null
  const tile = (label: string, value: string, note: string) =>
    h(
      'div',
      { class: 'bg-canvas px-6 py-6' },
      h('p', { class: 'text-3xl font-semibold tracking-[-0.03em] text-fg tabular-nums' }, value),
      h('p', { class: 'mt-1 text-[13px] text-body' }, label),
      h('p', { class: 'mt-0.5 text-xs text-muted' }, note),
    )
  const median = sp?.median ?? null
  return h(
    'div',
    { class: 'rule grid grid-cols-4 gap-px bg-line' },
    tile('Requests', a ? a.total.toLocaleString() : '–', 'since the server started'),
    tile('Per second', sp ? sp.perSecond.toFixed(sp.perSecond < 10 ? 1 : 0) : '–', 'over the last minute'),
    tile('Response time', median === null ? '–' : `${ms(median)} ms`, sp?.samples ? `median of the last ${sp.samples}` : 'no requests yet'),
    tile('Capacity', median === null ? '–' : `${Math.floor(1000 / median).toLocaleString()}/s`, 'one request at a time'),
  )
}

function server(url: string): HTMLElement {
  const row = (label: string, value: Node | string) =>
    h('div', { class: 'flex min-h-11 items-center justify-between gap-4 border-t border-line px-4 py-2 first:border-t-0' }, h('span', { class: 'text-[13px] text-muted' }, label), value)
  return band(
    'Server',
    h(
      'div',
      { class: `${card} mt-3` },
      row('Address', h('span', { class: 'flex items-center gap-1' }, h('span', { class: 'font-mono text-[13px] text-fg' }, url), copyButton(url, 'Copy the address'))),
      row(
        'Loaded models',
        state.loaded.length
          ? h('span', { class: 'flex flex-wrap justify-end gap-1.5' }, ...state.loaded.map((m) => h('span', { class: 'rounded-md bg-fill px-2 py-0.5 font-mono text-xs text-body' }, m)))
          : h('span', { class: 'text-[13px] text-muted' }, 'None. A model loads on its first request.'),
      ),
    ),
  )
}

function endpoints(url: string): HTMLElement {
  const list: [string, string, string][] = [
    ['POST', '/api/decide', 'Answers questions about a text or JSON, with a preset or your own questions'],
    ['POST', '/v1/decisions', 'TypeSafe-compatible: its SDKs work unchanged (also /v1/systemone)'],
    ['GET', '/v1/models', 'The models on this computer'],
  ]
  const model = [...state.installed].sort()[0] ?? 'largha'
  const example = `curl ${url}/api/decide -d '${JSON.stringify({ model: model.replace(/:latest$/, ''), preset: 'triage', state: 'I was charged twice this month.' })}'`
  return h(
    'section',
    { class: 'rule px-8 py-7', 'aria-labelledby': 'api-title' },
    h(
      'div',
      { class: 'flex items-baseline justify-between' },
      h('h2', { id: 'api-title', class: h2 }, 'API'),
      h('button', { class: 'inline-flex items-center gap-1 text-xs text-muted hover:text-fg', onClick: () => void backend.openLink('https://ollaya.dev/docs/api') }, 'Reference', icon(ICON.external, 'size-3.5')),
    ),
    h(
      'div',
      { class: `${card} mt-3` },
      ...list.map(([method, path, what]) =>
        h(
          'div',
          { class: 'flex min-h-11 items-center gap-3 border-t border-line px-4 py-2 first:border-t-0' },
          h('span', { class: 'w-11 font-mono text-[11px] font-semibold text-muted' }, method),
          h('span', { class: 'w-36 shrink-0 font-mono text-[13px] text-fg' }, path),
          h('span', { class: 'min-w-0 flex-1 truncate text-[13px] text-muted', title: what }, what),
          copyButton(url + path, `Copy ${path}`),
        ),
      ),
      h(
        'div',
        { class: 'flex items-start gap-2 rounded-b-md border-t border-line bg-code px-4 py-3' },
        h('code', { class: 'min-w-0 flex-1 font-mono text-xs leading-relaxed wrap-anywhere text-body' }, example),
        copyButton(example, 'Copy the example'),
      ),
    ),
  )
}

function recent(): HTMLElement {
  const a = state.activity
  const section = band('Recent requests')
  if (!a) {
    section.append(h('p', { class: 'mt-3 text-sm text-muted' }, state.activityError ? 'This server does not list its requests. Update Ollaya to 0.17.0 or newer.' : 'Loading…'))
    return section
  }
  if (!a.requests.length) {
    section.append(
      h(
        'div',
        { class: `${card} mt-3 px-4 py-6 text-center text-sm text-muted` },
        'No requests yet. ',
        h('button', { class: 'font-medium text-fg underline underline-offset-4', onClick: () => go({ page: 'models' }) }, 'Try a model'),
        ', or call the API from your code.',
      ),
    )
    return section
  }
  const cell = 'px-4 py-2'
  section.append(
    h(
      'div',
      { class: `${card} mt-3 overflow-hidden` },
      h(
        'table',
        { class: 'w-full text-[13px]' },
        h(
          'thead',
          { class: 'bg-subtle text-left text-xs text-muted' },
          h('tr', {}, ...['Time', 'Model', 'Endpoint', 'Questions', 'Took'].map((t, i) => h('th', { class: `${cell} font-medium ${i > 2 ? 'text-right' : ''}` }, t))),
        ),
        h(
          'tbody',
          {},
          ...a.requests.slice(0, 20).map((r) =>
            h(
              'tr',
              { class: 'border-t border-line' },
              h('td', { class: `${cell} text-muted tabular-nums` }, new Date(r.at).toLocaleTimeString()),
              h('td', { class: `${cell} font-mono text-xs text-fg` }, r.model),
              h('td', { class: `${cell} font-mono text-xs text-muted` }, r.endpoint),
              h('td', { class: `${cell} text-right tabular-nums ${r.error ? 'text-bad' : 'text-body'}` }, r.error ? r.error.toLowerCase().replaceAll('_', ' ') : String(r.questions)),
              h('td', { class: `${cell} text-right text-body tabular-nums` }, `${ms(r.total_duration / 1e6)} ms`),
            ),
          ),
        ),
      ),
    ),
  )
  return section
}

// --- view: models ---------------------------------------------------------------------------------

/** The Download button, or how far a download is, or that it is on this computer. */
function pullState(tag: string, compact: boolean): HTMLElement {
  const p = state.pulling.get(tag)
  if (p)
    return h(
      'span',
      { class: `flex items-center gap-2 ${compact ? 'w-24' : 'w-40'}` },
      h('span', { class: 'h-1.5 flex-1 overflow-hidden rounded-full bg-fill-strong' }, h('span', { class: 'bar block h-full rounded-full transition-[width]', style: `width:${p.total ? (p.completed / p.total) * 100 : 2}%` })),
      h('span', { class: 'w-8 text-right text-xs text-muted tabular-nums' }, percent(p)),
    )
  if (state.installed.has(tag)) return h('span', { class: 'flex items-center gap-1 text-xs text-muted' }, icon(ICON.check, 'size-4 text-ok'), compact ? 'On this computer' : 'Installed')
  return h(
    'button',
    {
      class: compact ? btnSmall : btnSecondary,
      disabled: !state.status?.running,
      title: state.status?.running ? undefined : 'Start the server first',
      onClick: (e: Event) => (e.stopPropagation(), void pull(tag)),
    },
    icon(ICON.download, 'size-3.5'),
    'Download',
  )
}

function models(): HTMLElement[] {
  const q = state.query.trim().toLowerCase()
  const shown = state.library
    .filter((m) => !state.onlyInstalled || installedTags(m).length)
    .filter((m) => !q || [m.name, m.title, m.description, m.publisher ?? '', ...m.caps].join(' ').toLowerCase().includes(q))
  const search = h('input', {
    class: `h-8 w-full pr-3 pl-8 text-[13px] ${field}`,
    type: 'search',
    placeholder: 'Search models',
    'aria-label': 'Search models',
    onInput: (e: Event) => {
      state.query = (e.target as HTMLInputElement).value
      render()
    },
  }) as HTMLInputElement
  search.value = state.query
  // The website's filter chips: an outline, filled when on.
  const tab = (label: string, only: boolean) =>
    h(
      'button',
      {
        class: `rounded-md border px-3 py-1 text-[13px] ${state.onlyInstalled === only ? 'border-fg bg-fg text-canvas' : 'border-line text-body hover:border-line-strong'}`,
        'aria-pressed': state.onlyInstalled === only ? 'true' : 'false',
        onClick: () => ((state.onlyInstalled = only), render()),
      },
      label,
    )
  const top = head(
    'Models',
    'Open decision models that run on this computer. Pick one to download and try it.',
    h(
      'div',
      { class: 'mt-6 flex items-center gap-3' },
      h('div', { class: 'relative w-64' }, icon(ICON.search, 'pointer-events-none absolute top-2 left-2.5 size-4 text-muted'), search),
      h('div', { class: 'flex gap-2' }, tab('All', false), tab('On this computer', true)),
    ),
    state.pullError ? h('p', { class: 'mt-3 text-xs text-bad' }, state.pullError) : null,
  )
  const note = (text: string) => band(null, h('p', { class: 'py-6 text-center text-sm text-muted' }, text))
  if (state.libraryError)
    return [
      top,
      band(
        null,
        h('p', { class: 'text-sm font-medium text-fg' }, 'The model library could not be reached'),
        h('p', { class: 'mt-1 text-sm text-muted' }, 'It comes from ollaya.dev. Check the internet connection and try again.'),
        h('p', { class: 'mt-2 text-xs break-all text-muted' }, state.libraryError),
        h('button', { class: `mt-4 ${btnSecondary}`, onClick: () => void loadLibrary() }, 'Try again'),
      ),
    ]
  if (!state.library.length) return [top, note('Loading the model library…')]
  if (!shown.length) return [top, note(state.onlyInstalled && !q ? 'No models on this computer yet.' : 'No model matches.')]
  // Empty cells close the last row, which would otherwise show the grid's line colour.
  const blank = (n: number, cls: string) => Array.from({ length: n }, () => h('li', { class: `bg-canvas ${cls}`, 'aria-hidden': 'true' }))
  return [
    top,
    h(
      'ul',
      { class: 'rule grid grid-cols-2 gap-px border-b border-line bg-line lg:grid-cols-3', role: 'list' },
      ...shown.map(modelCard),
      ...blank((2 - (shown.length % 2)) % 2, 'lg:hidden'),
      ...blank((3 - (shown.length % 3)) % 3, 'hidden lg:block'),
    ),
  ]
}

/** A model as the website's card: who made it, what it is, its numbers from the bench. */
function modelCard(m: LibraryModel): HTMLElement {
  const latest = `${m.name}:latest`
  const have = installedTags(m)
  const size = tagOf(m, latest)?.size
  const summary = (m.description.match(/^.+?[.!?](?=\s|$)/)?.[0] ?? m.description).replace(/\.$/, '')
  const stat = (value: string, label: string) =>
    h('div', {}, h('p', { class: 'text-lg font-semibold tracking-[-0.02em] text-fg tabular-nums' }, value), h('p', { class: 'text-[11px] text-muted' }, label))
  const open = () => go({ page: 'model', name: m.name })
  return h(
    'li',
    { class: 'flex' },
    h(
      'div',
      {
        class: 'group flex w-full cursor-default flex-col bg-canvas p-5 hover:bg-raised',
        role: 'button',
        tabindex: '0',
        'aria-label': `${title(m)}: open`,
        onClick: open,
        onKeydown: (e: Event) => {
          if ((e as KeyboardEvent).key === 'Enter') open()
        },
      },
      h(
        'div',
        { class: 'flex items-start justify-between gap-3' },
        h(
          'div',
          { class: 'min-w-0' },
          h('p', { class: 'truncate font-mono text-[15px] text-fg underline-offset-4 group-hover:underline' }, m.name),
          h('p', { class: 'mt-0.5 truncate text-xs text-muted' }, m.publisher ?? ''),
        ),
        have.length && !state.pulling.has(latest) ? pullState(have[0]!, true) : pullState(latest, true),
      ),
      h('p', { class: 'mt-3 line-clamp-3 flex-1 text-[13px] leading-5 text-body' }, summary),
      h(
        'div',
        { class: 'mt-5 flex items-end gap-5 border-t border-line pt-3.5' },
        stat(m.stats?.accuracy ? m.stats.accuracy.toFixed(3) : '—', 'accuracy'),
        stat(m.stats?.latencyMs ? `${Math.round(m.stats.latencyMs)} ms` : '—', 'five questions'),
        h('div', { class: 'ml-auto text-right text-xs text-muted tabular-nums' }, size ? bytes(size) : ''),
      ),
    ),
  )
}

// --- view: one model ------------------------------------------------------------------------------

function modelPage(m: LibraryModel | undefined): HTMLElement[] {
  if (!m) return [h('p', { class: 'pt-24 text-center text-sm text-muted' }, 'Loading the model library…')]
  const running = !!state.status?.running
  const bands = [
    h(
      'header',
      { class: 'px-8 pt-8 pb-8' },
      h(
        'nav',
        { class: 'flex items-center gap-1.5 text-[13px] text-muted', 'aria-label': 'Breadcrumb' },
        h('button', { class: 'underline-offset-4 hover:text-fg hover:underline', onClick: () => go({ page: 'models' }) }, 'Models'),
        h('span', { 'aria-hidden': 'true' }, '/'),
        h('span', { class: 'text-body' }, m.name),
      ),
      h(
        'div',
        { class: 'mt-4 flex items-start justify-between gap-4' },
        h(
          'div',
          { class: 'min-w-0' },
          h('h1', { class: 'truncate font-mono text-3xl font-semibold tracking-tight text-fg' }, m.name),
          h('p', { class: 'mt-1.5 text-[13px] text-muted' }, [m.publisher && `by ${m.publisher}`, m.license].filter(Boolean).join(' · ')),
        ),
        pullState(`${m.name}:latest`, false),
      ),
      h('p', { class: 'mt-4 max-w-2xl text-[15px] text-body' }, m.description),
      h('div', { class: 'mt-4 flex flex-wrap gap-2' }, ...m.caps.map((c) => h('span', { class: 'rounded-md bg-cap-bg px-2 py-[2px] text-[13px] font-medium text-cap-fg' }, c))),
      state.pullError ? h('p', { class: 'mt-3 text-xs text-bad' }, state.pullError) : null,
    ),
  ]
  if (!running) bands.push(band(null, h('p', { class: 'text-sm text-muted' }, 'Start the server to download and try models.')))
  else if (installedTags(m).length) bands.push(tryPanel(m))
  bands.push(band('Versions', h('div', { class: `${card} mt-3 overflow-hidden` }, ...mainTags(m).map((t, i) => versionRow(m, t, i)))))
  if (m.readme) bands.push(readme(m))
  bands.push(
    band(
      null,
      h(
        'button',
        { class: 'inline-flex items-center gap-1 text-xs text-muted hover:text-fg', onClick: () => void backend.openLink(`https://ollaya.dev/library/${m.name}`) },
        `${m.name} on ollaya.dev`,
        icon(ICON.external, 'size-3.5'),
      ),
    ),
  )
  return bands
}

function versionRow(m: LibraryModel, tag: string, i: number): HTMLElement {
  const t = tagOf(m, tag)
  const summary = t?.summary || (tag.endsWith(':latest') ? `The default version: ollaya run ${m.name}` : '')
  const have = state.installed.has(tag) && !state.pulling.has(tag)
  return h(
    'div',
    { class: `flex min-h-14 items-center justify-between gap-4 px-4 py-2.5 ${i ? 'border-t border-line' : ''}` },
    h(
      'div',
      { class: 'min-w-0' },
      h('p', { class: 'font-mono text-[13px] text-fg' }, tag, t?.size ? h('span', { class: 'ml-2 font-sans text-xs text-muted' }, bytes(t.size)) : null),
      summary ? h('p', { class: 'mt-0.5 truncate text-xs text-muted', title: summary }, summary) : null,
    ),
    h(
      'span',
      { class: 'flex shrink-0 items-center gap-1' },
      pullState(tag, false),
      have
        ? h('button', { class: 'ml-1 inline-flex size-7 items-center justify-center rounded-md text-muted hover:bg-fill hover:text-fg', title: `Remove ${tag}`, 'aria-label': `Remove ${tag}`, onClick: () => void remove(tag) }, icon(ICON.x, 'size-4'))
        : null,
    ),
  )
}

/** The model card from ollaya.dev. Its links open in the browser, never inside the app. */
function readme(m: LibraryModel): HTMLElement {
  const body = h('div', { class: 'prose mt-4' })
  body.innerHTML = m.readme! // ollaya.dev's own HTML; the page's CSP runs none of its scripts
  body.addEventListener('click', (e) => {
    const a = (e.target as Element).closest('a')
    if (!a) return
    e.preventDefault()
    const href = a.getAttribute('href') ?? ''
    if (href.startsWith('#')) body.querySelector(href)?.scrollIntoView({ behavior: 'smooth' })
    else void backend.openLink(new URL(href, `https://ollaya.dev/library/${m.name}`).href)
  })
  return band('Model card', body)
}

function tryPanel(m: LibraryModel): HTMLElement {
  const tags = installedTags(m)
  if (!tags.includes(state.runModel)) state.runModel = tags[0] ?? ''
  // A model with built-in questions takes no preset or custom questions. Until the server says
  // which kind this version is, neither the inputs nor the note show.
  const own = state.builtin.get(state.runModel)
  if (own === undefined && state.runModel) void builtinOf(state.runModel)
  const selectCls = `h-8 px-2.5 text-[13px] ${field}`
  const textarea = h('textarea', {
    class: `mt-3 block h-28 w-full resize-y px-3.5 py-3 text-sm leading-relaxed ${field}`,
    placeholder: 'Paste a message, an email or a JSON object…',
    spellcheck: 'false',
    onInput: (e) => {
      state.input = (e.target as HTMLTextAreaElement).value
      // Enable Run as soon as there is text, without redrawing the page under the caret.
      const button = document.getElementById('run-button') as HTMLButtonElement | null
      if (button) button.disabled = state.running || !state.input.trim()
    },
    onKeydown: (e) => {
      const k = e as KeyboardEvent
      if (k.key === 'Enter' && (k.metaKey || k.ctrlKey)) void run()
    },
  }) as HTMLTextAreaElement
  textarea.value = state.input
  const panel = band(
    'Try it',
    h(
      'div',
      { class: 'mt-3 flex flex-wrap items-center gap-2' },
      tags.length > 1
        ? labelled('Version', h('select', { class: selectCls, onChange: (e) => ((state.runModel = (e.target as HTMLSelectElement).value), render()) }, ...tags.map((t) => option(t, t, t === state.runModel))))
        : null,
      own
        ? h('p', { class: 'text-[13px] text-muted' }, 'This model asks its own questions: ', h('span', { class: 'font-mono text-xs text-body' }, own.join(', ')))
        : own === null
          ? labelled(
              'Questions',
              h(
                'select',
                { class: selectCls, onChange: (e) => ((state.preset = (e.target as HTMLSelectElement).value), render()) },
                ...state.presets.map((p) => option(p.name, p.builtin ? `${p.name} preset` : `${p.name} (your preset)`, p.name === state.preset)),
                option('custom', 'Custom JSON', state.preset === 'custom'),
              ),
            )
          : null,
    ),
    textarea,
  )
  if (own === null && state.preset === 'custom') {
    const q = h('textarea', {
      class: `mt-2 block h-40 w-full resize-y px-3.5 py-3 font-mono text-xs leading-relaxed ${field}`,
      spellcheck: 'false',
      'aria-label': 'Questions as JSON',
      onInput: (e) => (state.customQuestions = (e.target as HTMLTextAreaElement).value),
    }) as HTMLTextAreaElement
    q.value = state.customQuestions
    panel.append(q)
  }
  const mac = navigator.platform.toLowerCase().includes('mac')
  panel.append(
    h(
      'div',
      { class: 'mt-3 flex items-center justify-between gap-3' },
      h('button', { class: 'text-xs text-muted underline underline-offset-4 hover:text-fg', onClick: () => ((state.input = EXAMPLE), render()) }, 'Use an example'),
      h(
        'span',
        { class: 'flex items-center gap-3' },
        h('span', { class: 'text-xs text-muted' }, mac ? '⌘ Return' : 'Ctrl + Enter'),
        h('button', { id: 'run-button', class: btnPrimary, disabled: state.running || !state.input.trim(), onClick: () => void run() }, state.running ? icon(ICON.spinner, 'size-3.5') : null, 'Run'),
      ),
    ),
  )
  if (state.runError) panel.append(h('p', { class: 'mt-4 text-sm text-bad' }, state.runError))
  if (state.result) panel.append(results(state.result))
  return panel
}

function labelled(label: string, control: HTMLElement): HTMLElement {
  return h('label', { class: 'flex items-center gap-2 text-[13px] text-muted' }, label, control)
}

function option(value: string, text: string, selected: boolean): HTMLElement {
  return h('option', { value, selected }, text)
}

/** One row per answer, as the website's hero shows them: the question, the answer, a meter of ten
 * segments and its probability. */
function results(r: DecideResponse): HTMLElement {
  const rows = Object.entries(r.answers).map(([q, a]) => {
    const [text, note, p] = describe(a)
    const lit = Math.round(p * 10)
    return h(
      'li',
      { class: 'border-t border-line px-5 py-3 first:border-t-0' },
      h('p', { class: 'font-mono text-xs text-muted' }, q),
      h(
        'div',
        { class: 'mt-1.5 flex items-center gap-4' },
        h('span', { class: 'w-48 shrink-0 truncate text-[15px] font-medium text-fg', title: [text, note].filter(Boolean).join(': ') }, text, note ? h('span', { class: 'ml-2 text-[13px] font-normal text-muted' }, note) : null),
        h('span', { class: 'meter min-w-0 flex-1', 'aria-hidden': 'true' }, ...Array.from({ length: 10 }, (_, i) => h('i', { class: i < lit ? 'on' : undefined }))),
        h('span', { class: 'w-10 text-right text-[13px] text-fg tabular-nums' }, p.toFixed(2)),
      ),
    )
  })
  return h(
    'div',
    { id: 'answers', class: `${card} mt-6 overflow-hidden bg-subtle` },
    h('ul', { role: 'list' }, ...rows),
    h(
      'p',
      { class: 'border-t border-line px-5 py-3 text-xs text-muted' },
      'Answered by ',
      h('span', { class: 'font-mono text-body' }, r.routing?.model ?? r.model),
      ' in ',
      h('span', { class: 'font-medium text-fg' }, `${ms(r.total_duration / 1e6)} ms`),
    ),
  )
}

function describe(a: Answer): [string, string, number] {
  switch (a.type) {
    case 'choice':
      return [a.choice ?? '', '', a.confidence ?? 0]
    case 'score': {
      const levels = Object.keys(a.legend ?? {}).length - 1
      const near = a.legend?.[String(Math.round(a.score ?? 0))] ?? ''
      return [`${(a.score ?? 0).toFixed(2)} / ${levels}`, near, a.confidence ?? 0]
    }
    case 'noul': {
      const p = a.noul ?? 0
      return [p >= 0.5 ? 'yes' : 'no', '', p >= 0.5 ? p : 1 - p]
    }
  }
}

// --- render loop ----------------------------------------------------------------------------------

const root = document.getElementById('app')!

function render() {
  // Keep the focused field's caret across redraws.
  const active = document.activeElement as HTMLInputElement | HTMLTextAreaElement | null
  const field = active && ['TEXTAREA', 'INPUT'].includes(active.tagName) ? (active.getAttribute('aria-label') ?? 'state') : null
  const caret = field ? ([active!.selectionStart, active!.selectionEnd] as const) : null
  const scrolled = document.getElementById('main-pane')?.scrollTop ?? 0
  root.replaceChildren(header(), h('div', { class: 'flex min-h-0 flex-1' }, sidebar(), main()))
  document.getElementById('main-pane')?.scrollTo({ top: scrolled })
  if (field && caret) {
    const again = [...root.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>('textarea, input')].find((t) => (t.getAttribute('aria-label') ?? 'state') === field)
    again?.focus()
    again?.setSelectionRange(caret[0], caret[1])
  }
}

backend.onPullProgress((p) => {
  if (p.done) return
  state.pulling.set(p.model, { completed: p.completed, total: p.total, status: p.status })
  render()
})

async function loadLibrary() {
  state.libraryError = ''
  render()
  try {
    state.library = [...(await backend.library())].sort((a, b) => a.rank - b.rank)
  } catch (e) {
    state.libraryError = String(e)
  }
  render()
}

async function boot() {
  render()
  const [, presets] = await Promise.allSettled([loadLibrary(), backend.presets()])
  if (presets.status === 'fulfilled') {
    state.presets = presets.value
    state.customQuestions = JSON.stringify(presets.value.find((p) => p.name === 'triage')?.questions ?? {}, null, 2)
  }
  await refresh()
  // Like Ollama's app: the server runs while the app is open.
  if (!state.status?.running) await toggleServer()
  // A first run has nothing to show on the overview: start where the models are.
  if (!state.installed.size) state.view = { page: 'models' }
  // Custom presets live on the server, which may only have started just now.
  state.presets = await backend.presets().catch(() => state.presets)
  state.update = await backend.updateAvailable().catch(() => null)
  render()
  setInterval(() => void refresh(), 2000)
  // Preview only (npm run preview): #result shows a finished run, #pull a download in progress.
  if (__MOCK__ && location.hash === '#result') {
    go({ page: 'model', name: 'laya' })
    state.input = EXAMPLE
    await run()
  }
  if (__MOCK__ && location.hash === '#pull') void pull('decider:latest'), go({ page: 'model', name: 'decider' })
}

declare const __MOCK__: boolean

void boot()
