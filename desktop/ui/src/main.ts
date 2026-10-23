// Ollaya desktop, laid out as a desktop app: a rail of pages on the left (Discover, My Models,
// Playground, Server), the page in the middle and the server's status in a bar along the bottom.
// Plain DOM, rendered from one state object.
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
  external: svg('<path d="M11 4h5v5m0-5-7 7M8 5H5a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-3"/>'),
  copy: svg('<rect x="7" y="7" width="9" height="9" rx="1.5"/><path d="M13 7V5.5A1.5 1.5 0 0 0 11.5 4h-6A1.5 1.5 0 0 0 4 5.5v6A1.5 1.5 0 0 0 5.5 13H7"/>'),
  discover: svg('<circle cx="10" cy="10" r="7"/><path d="m12.8 7.2-1.6 4-4 1.6 1.6-4z"/>'),
  models: svg('<path d="M3.5 6.5 10 3l6.5 3.5v7L10 17l-6.5-3.5zM3.5 6.5 10 10l6.5-3.5M10 10v7"/>'),
  play: svg('<path d="M6.5 4.5v11l9-5.5z"/>'),
  server: svg('<path d="M3 10h3l2-5 4 10 2-5h3"/>'),
  search: svg('<circle cx="9" cy="9" r="5"/><path d="m16 16-3.5-3.5"/>'),
  chevron: svg('<path d="m6 8 4 4 4-4"/>'),
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

type Page = 'discover' | 'mine' | 'playground' | 'server'

const state = {
  page: 'discover' as Page,
  status: null as Status | null,
  statusBusy: false,
  statusError: '',
  /** A newer Ollaya release than this app. */
  update: null as string | null,
  library: [] as LibraryModel[],
  libraryError: '',
  /** The versions on this computer, with their size. */
  installed: new Map<string, number>(),
  loaded: [] as string[],
  activity: null as ActivityResponse | null,
  activityError: '',
  query: '',
  onlyInstalled: false,
  /** The family open in Discover's detail pane. */
  selected: '',
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
/** Every installed version: the library's order first, then what the library does not list. */
const allInstalled = () => {
  const known = state.library.flatMap(installedTags)
  return [...known, ...[...state.installed.keys()].filter((t) => !known.includes(t)).sort()]
}

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
  const snapshot = () => JSON.stringify([state.status, [...state.installed], state.loaded, state.activity, state.activityError])
  const before = snapshot()
  state.status = await backend.status()
  if (state.status.running) {
    const [installed, loaded, activity] = await Promise.allSettled([backend.installed(), backend.loaded(), backend.activity()])
    state.installed = new Map(installed.status === 'fulfilled' ? installed.value.map((m) => [m.name, m.size]) : [])
    state.loaded = loaded.status === 'fulfilled' ? loaded.value : []
    state.activity = activity.status === 'fulfilled' ? activity.value : null
    // A server older than 0.17.0 (a command-line install) has no activity log.
    state.activityError = activity.status === 'rejected' ? String(activity.reason) : ''
  } else {
    state.installed = new Map()
    state.loaded = []
    state.activity = null
  }
  // The server page has no field to keep, and its per-second rate moves with the clock.
  if (state.page === 'server' || snapshot() !== before) render()
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
}

function go(page: Page) {
  state.page = page
  render()
}

/** Open the playground with `tag` picked. */
function tryOut(tag: string) {
  state.runModel = tag
  state.result = null
  state.runError = ''
  go('playground')
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

// --- view: pieces ---------------------------------------------------------------------------------

const btnPrimary =
  'inline-flex h-7 items-center justify-center gap-1.5 rounded-md bg-btn px-3 text-xs font-medium text-btn-fg hover:bg-btn-hover disabled:opacity-50'
const btnSecondary =
  'inline-flex h-7 items-center justify-center gap-1.5 rounded-md border border-line px-3 text-xs font-medium text-fg hover:border-line-strong hover:bg-fill disabled:opacity-50'
const btnQuiet = 'inline-flex size-7 shrink-0 items-center justify-center rounded-md text-muted hover:bg-fill hover:text-fg disabled:opacity-50'
const field = 'rounded-md border border-line bg-canvas text-fg placeholder:text-muted hover:border-line-strong focus:border-line-strong focus:outline-none'
const panel = 'rounded-md border border-line bg-panel'
/** A section's label inside a page. */
const label = 'text-[11px] font-medium tracking-[0.06em] text-muted uppercase'
const mono = 'font-mono text-xs'

/** What the server's device means, for the status bar's tooltip (#44). */
function deviceHint(device: string | null): string {
  if (device === 'CPU only')
    return "The app's built-in server runs every model on the CPU. Install the command-line Ollaya with its GPU pack (install.ps1 or install.sh) and the app starts the server from there, on your NVIDIA GPU."
  if (device === 'GPU') return 'The server runs from the command-line install, which has the GPU pack.'
  return ''
}

function copyButton(text: string, what: string): HTMLElement {
  const done = state.copied === text
  return h(
    'button',
    { class: btnQuiet, title: done ? 'Copied' : what, 'aria-label': what, onClick: () => void copy(text) },
    icon(done ? ICON.check : ICON.copy, `size-4 ${done ? 'text-ok' : ''}`),
  )
}

function select(attrs: Attrs, ...options: HTMLElement[]): HTMLElement {
  return h(
    'span',
    { class: 'relative inline-flex' },
    h('select', { ...attrs, class: `h-7 appearance-none pr-7 pl-2.5 text-xs ${field}` }, ...options),
    icon(ICON.chevron, 'pointer-events-none absolute top-1.5 right-2 size-3.5 text-muted'),
  )
}

function option(value: string, text: string, selected: boolean): HTMLElement {
  return h('option', { value, selected }, text)
}

/** The Download button, or how far a download is, or that it is on this computer. */
function pullState(tag: string): HTMLElement {
  const p = state.pulling.get(tag)
  if (p)
    return h(
      'span',
      { class: 'flex w-36 items-center gap-2' },
      h('span', { class: 'h-1 flex-1 overflow-hidden rounded-full bg-fill-strong' }, h('span', { class: 'block h-full rounded-full bg-us transition-[width]', style: `width:${p.total ? (p.completed / p.total) * 100 : 2}%` })),
      h('span', { class: 'w-8 text-right text-[11px] text-muted tabular-nums' }, percent(p)),
    )
  if (state.installed.has(tag)) return h('span', { class: 'inline-flex items-center gap-1 text-xs text-muted' }, icon(ICON.check, 'size-3.5 text-ok'), 'Installed')
  return h(
    'button',
    {
      class: btnSecondary,
      disabled: !state.status?.running,
      title: state.status?.running ? undefined : 'Start the server first',
      onClick: () => void pull(tag),
    },
    icon(ICON.download, 'size-3.5'),
    'Download',
  )
}

function empty(text: string, ...rest: Child[]): HTMLElement {
  return h('div', { class: 'flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center text-[13px] text-muted' }, h('p', {}, text), ...rest)
}

// --- view: the rail and the status bar ------------------------------------------------------------

function rail(): HTMLElement {
  const item = (page: Page, text: string, svgIcon: string, note?: string) =>
    h(
      'button',
      {
        class: `relative flex w-full flex-col items-center gap-1 rounded-md py-2 text-[10px] ${state.page === page ? 'bg-fill text-fg' : 'text-muted hover:bg-fill hover:text-fg'}`,
        'aria-current': state.page === page ? 'page' : undefined,
        onClick: () => go(page),
      },
      icon(svgIcon, 'size-5'),
      text,
      note ? h('span', { class: 'absolute top-1 right-2 rounded-full bg-fill-strong px-1.5 text-[9px] leading-4 text-body tabular-nums' }, note) : null,
    )
  const have = state.installed.size
  return h(
    'nav',
    { class: 'flex w-[72px] shrink-0 flex-col items-center gap-1 border-r border-line px-2 pt-3', 'aria-label': 'Pages' },
    icon(GAZE, 'mb-3 size-7 text-fg'),
    item('discover', 'Discover', ICON.discover),
    item('mine', 'My Models', ICON.models, have ? String(have) : undefined),
    item('playground', 'Playground', ICON.play),
    item('server', 'Server', ICON.server),
  )
}

function statusBar(): HTMLElement {
  const s = state.status
  const running = !!s?.running
  const dot = h('span', { class: `size-2 rounded-full ${running ? 'bg-ok' : 'bg-faint'}` })
  const what = s === null ? 'Checking the server…' : running ? `Running on ${s.url.replace(/^https?:\/\//, '')}` : 'Server stopped'
  return h(
    'footer',
    { class: 'flex h-8 shrink-0 items-center gap-3 border-t border-line px-3 text-xs text-muted' },
    h('span', { class: 'flex items-center gap-2', title: running ? deviceHint(s?.device ?? null) : '' }, dot, h('span', { class: running ? 'text-body' : '' }, what), running && s?.device ? h('span', {}, `· ${s.device}`) : null),
    running ? h('span', { class: 'truncate' }, state.loaded.length ? `· loaded: ${state.loaded.join(', ')}` : '· nothing loaded') : null,
    state.statusError ? h('span', { class: 'truncate text-bad', title: state.statusError }, state.statusError) : null,
    h('span', { class: 'flex-1' }),
    state.update ? h('button', { class: 'text-fg underline underline-offset-4', title: 'Opens the download page', onClick: () => void backend.openDownload() }, `Update to ${state.update}`) : null,
    h(
      'button',
      { class: `${btnSecondary} h-6`, disabled: state.statusBusy || s === null, onClick: toggleServer },
      state.statusBusy ? icon(ICON.spinner, 'size-3') : null,
      running ? 'Stop' : 'Start',
    ),
    s?.version ? h('span', { class: 'tabular-nums' }, `v${s.version}`) : null,
  )
}

// --- view: discover -------------------------------------------------------------------------------

function discover(): HTMLElement {
  const q = state.query.trim().toLowerCase()
  const shown = state.library
    .filter((m) => !state.onlyInstalled || installedTags(m).length)
    .filter((m) => !q || [m.name, m.title, m.description, m.publisher ?? '', ...m.caps].join(' ').toLowerCase().includes(q))
  const search = h('input', {
    class: `h-7 w-full pr-2 pl-7 text-xs ${field}`,
    type: 'search',
    placeholder: 'Search models',
    'aria-label': 'Search models',
    onInput: (e: Event) => {
      state.query = (e.target as HTMLInputElement).value
      render()
    },
  }) as HTMLInputElement
  search.value = state.query
  const tab = (text: string, only: boolean) =>
    h(
      'button',
      { class: `rounded px-2 py-0.5 ${state.onlyInstalled === only ? 'bg-fill-strong text-fg' : 'text-muted hover:text-fg'}`, 'aria-pressed': state.onlyInstalled === only ? 'true' : 'false', onClick: () => ((state.onlyInstalled = only), render()) },
      text,
    )
  const m = modelNamed(state.selected) ?? shown[0]
  const list = h(
    'aside',
    { class: 'flex w-72 shrink-0 flex-col border-r border-line' },
    h(
      'div',
      { class: 'space-y-2 border-b border-line p-2.5' },
      h('div', { class: 'relative' }, icon(ICON.search, 'pointer-events-none absolute top-1.5 left-2 size-4 text-muted'), search),
      h('div', { class: 'flex items-center gap-1 text-[11px]' }, tab('All', false), tab('Installed', true), h('span', { class: 'ml-auto text-muted tabular-nums' }, state.library.length ? `${shown.length} of ${state.library.length}` : '')),
    ),
    state.libraryError
      ? empty('The model library could not be reached. It comes from ollaya.dev.', h('button', { class: btnSecondary, onClick: () => void loadLibrary() }, 'Try again'))
      : !state.library.length
        ? empty('Loading the model library…')
        : !shown.length
          ? empty(state.onlyInstalled && !q ? 'No models on this computer yet.' : 'No model matches.')
          : h('ul', { id: 'list-pane', class: 'min-h-0 flex-1 overflow-y-auto p-1.5', role: 'list' }, ...shown.map((x) => modelRow(x, x === m))),
  )
  return h('div', { class: 'flex min-h-0 flex-1' }, list, m ? detail(m) : h('div', { class: 'flex flex-1' }, empty('Pick a model on the left.')))
}

function modelRow(m: LibraryModel, on: boolean): HTMLElement {
  const have = installedTags(m).length
  const size = tagOf(m, `${m.name}:latest`)?.size
  return h(
    'li',
    {},
    h(
      'button',
      {
        class: `flex w-full items-center gap-3 rounded-md px-2.5 py-2 text-left ${on ? 'bg-fill text-fg' : 'hover:bg-fill'}`,
        'aria-current': on ? 'true' : undefined,
        onClick: () => ((state.selected = m.name), render()),
      },
      h(
        'span',
        { class: 'min-w-0 flex-1' },
        h('span', { class: `block truncate ${mono} text-[13px] text-fg` }, m.name),
        h('span', { class: 'block truncate text-[11px] text-muted' }, m.publisher ?? m.title),
      ),
      h('span', { class: 'flex shrink-0 flex-col items-end gap-0.5 text-[11px] text-muted tabular-nums' }, size ? bytes(size) : '', have ? icon(ICON.check, 'size-3.5 text-ok') : null),
    ),
  )
}

function detail(m: LibraryModel): HTMLElement {
  const running = !!state.status?.running
  const stat = (value: string, what: string) => h('span', {}, h('span', { class: 'font-medium text-fg tabular-nums' }, value), ` ${what}`)
  const have = installedTags(m)
  return h(
    'div',
    { id: 'main-pane', class: 'min-w-0 flex-1 overflow-y-auto' },
    h(
      'div',
      { class: 'mx-auto max-w-3xl p-6' },
      h(
        'div',
        { class: 'flex items-start justify-between gap-4' },
        h(
          'div',
          { class: 'min-w-0' },
          h('h1', { class: 'font-mono text-xl font-semibold tracking-tight text-fg' }, m.name),
          h('p', { class: 'mt-1 text-xs text-muted' }, [m.publisher && `by ${m.publisher}`, m.license].filter(Boolean).join(' · ')),
        ),
        h('span', { class: 'flex shrink-0 items-center gap-2' }, have.length ? h('button', { class: btnPrimary, disabled: !running, onClick: () => tryOut(have[0]!) }, icon(ICON.play, 'size-3.5'), 'Try') : null, pullState(`${m.name}:latest`)),
      ),
      h('p', { class: 'mt-4 text-[13px] leading-relaxed text-body' }, m.description),
      h(
        'p',
        { class: 'mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted' },
        m.stats?.accuracy ? stat(m.stats.accuracy.toFixed(3), 'accuracy') : null,
        m.stats?.latencyMs ? stat(`${Math.round(m.stats.latencyMs)} ms`, 'for five questions') : null,
        ...m.caps.map((c) => h('span', { class: 'rounded bg-cap-bg px-1.5 py-px text-[11px] font-medium text-cap-fg' }, c)),
      ),
      state.pullError ? h('p', { class: 'mt-3 text-xs text-bad' }, state.pullError) : null,
      h('p', { class: `${label} mt-8` }, 'Versions'),
      h('div', { class: `${panel} mt-2 overflow-hidden` }, ...mainTags(m).map((t, i) => versionRow(m, t, i))),
      m.readme ? readme(m) : null,
      h(
        'button',
        { class: 'mt-8 inline-flex items-center gap-1 text-xs text-muted hover:text-fg', onClick: () => void backend.openLink(`https://ollaya.dev/library/${m.name}`) },
        `${m.name} on ollaya.dev`,
        icon(ICON.external, 'size-3.5'),
      ),
    ),
  )
}

function versionRow(m: LibraryModel, tag: string, i: number): HTMLElement {
  const t = tagOf(m, tag)
  const summary = t?.summary || (tag.endsWith(':latest') ? `The default version: ollaya run ${m.name}` : '')
  const have = state.installed.has(tag) && !state.pulling.has(tag)
  return h(
    'div',
    { class: `flex min-h-11 items-center justify-between gap-4 px-3 py-2 ${i ? 'border-t border-line' : ''}` },
    h(
      'div',
      { class: 'min-w-0' },
      h('p', { class: `${mono} text-fg` }, tag, t?.size ? h('span', { class: 'ml-2 font-sans text-[11px] text-muted' }, bytes(t.size)) : null),
      summary ? h('p', { class: 'mt-0.5 truncate text-[11px] text-muted', title: summary }, summary) : null,
    ),
    h(
      'span',
      { class: 'flex shrink-0 items-center gap-1' },
      have ? h('button', { class: btnQuiet, title: `Try ${tag}`, 'aria-label': `Try ${tag}`, disabled: !state.status?.running, onClick: () => tryOut(tag) }, icon(ICON.play, 'size-4')) : null,
      pullState(tag),
      have ? h('button', { class: btnQuiet, title: `Remove ${tag}`, 'aria-label': `Remove ${tag}`, onClick: () => void remove(tag) }, icon(ICON.x, 'size-4')) : null,
    ),
  )
}

/** The model card from ollaya.dev. Its links open in the browser, never inside the app. */
function readme(m: LibraryModel): HTMLElement {
  const body = h('div', { class: 'prose mt-2' })
  body.innerHTML = m.readme! // ollaya.dev's own HTML; the page's CSP runs none of its scripts
  body.addEventListener('click', (e) => {
    const a = (e.target as Element).closest('a')
    if (!a) return
    e.preventDefault()
    const href = a.getAttribute('href') ?? ''
    if (href.startsWith('#')) body.querySelector(href)?.scrollIntoView({ behavior: 'smooth' })
    else void backend.openLink(new URL(href, `https://ollaya.dev/library/${m.name}`).href)
  })
  return h('section', { 'aria-label': 'Model card' }, h('p', { class: `${label} mt-8` }, 'Model card'), body)
}

// --- view: my models ------------------------------------------------------------------------------

function mine(): HTMLElement {
  const tags = allInstalled()
  const running = !!state.status?.running
  if (!running) return empty(state.status === null ? 'Checking the server…' : 'Start the server to see the models on this computer.')
  if (!tags.length) return empty('No models on this computer yet.', h('button', { class: btnSecondary, onClick: () => go('discover') }, 'Discover models'))
  const cell = 'px-3 py-2'
  const total = tags.reduce((n, t) => n + (state.installed.get(t) ?? 0), 0)
  return h(
    'div',
    { id: 'main-pane', class: 'min-w-0 flex-1 overflow-y-auto' },
    h(
      'div',
      { class: 'mx-auto max-w-3xl p-6' },
      h('div', { class: 'flex items-baseline justify-between' }, h('p', { class: label }, 'On this computer'), h('p', { class: 'text-xs text-muted tabular-nums' }, `${tags.length} ${tags.length === 1 ? 'version' : 'versions'} · ${bytes(total)}`)),
      state.pullError ? h('p', { class: 'mt-3 text-xs text-bad' }, state.pullError) : null,
      h(
        'div',
        { class: `${panel} mt-2 overflow-hidden` },
        h(
          'table',
          { class: 'w-full text-xs' },
          h('thead', { class: 'text-left text-[11px] text-muted' }, h('tr', {}, ...['Version', 'Size', 'State', ''].map((t) => h('th', { class: `${cell} font-medium` }, t)))),
          h(
            'tbody',
            {},
            ...tags.map((t) =>
              h(
                'tr',
                { class: 'border-t border-line' },
                h('td', { class: `${cell} ${mono} text-fg` }, t),
                h('td', { class: `${cell} text-muted tabular-nums` }, bytes(state.installed.get(t) ?? 0)),
                h('td', { class: `${cell} text-muted` }, state.loaded.includes(t) ? h('span', { class: 'inline-flex items-center gap-1.5 text-body' }, h('span', { class: 'size-1.5 rounded-full bg-ok' }), 'Loaded') : 'On disk'),
                h(
                  'td',
                  { class: `${cell} text-right` },
                  h('span', { class: 'inline-flex gap-1' }, h('button', { class: btnQuiet, title: `Try ${t}`, 'aria-label': `Try ${t}`, onClick: () => tryOut(t) }, icon(ICON.play, 'size-4')), h('button', { class: btnQuiet, title: `Remove ${t}`, 'aria-label': `Remove ${t}`, onClick: () => void remove(t) }, icon(ICON.x, 'size-4'))),
                ),
              ),
            ),
          ),
        ),
      ),
    ),
  )
}

// --- view: playground -----------------------------------------------------------------------------

function playground(): HTMLElement {
  const tags = allInstalled()
  const running = !!state.status?.running
  if (!running) return empty(state.status === null ? 'Checking the server…' : 'Start the server to run a model.')
  if (!tags.length) return empty('Download a model first.', h('button', { class: btnSecondary, onClick: () => go('discover') }, 'Discover models'))
  if (!tags.includes(state.runModel)) state.runModel = tags[0]!
  // A model with built-in questions takes no preset or custom questions. Until the server says
  // which kind this version is, neither the inputs nor the note show.
  const own = state.builtin.get(state.runModel)
  if (own === undefined) void builtinOf(state.runModel)
  const mac = navigator.platform.toLowerCase().includes('mac')
  const toolbar = h(
    'div',
    { class: 'flex shrink-0 flex-wrap items-center gap-2 border-b border-line px-4 py-2 text-xs text-muted' },
    'Model',
    select({ 'aria-label': 'Model', onChange: (e) => ((state.runModel = (e.target as HTMLSelectElement).value), (state.result = null), render()) }, ...tags.map((t) => option(t, t, t === state.runModel))),
    own
      ? h('span', { class: 'ml-2' }, 'Asks its own questions: ', h('span', { class: `${mono} text-body` }, own.join(', ')))
      : own === null
        ? h(
            'span',
            { class: 'ml-2 inline-flex items-center gap-2' },
            'Questions',
            select(
              { 'aria-label': 'Questions', onChange: (e) => ((state.preset = (e.target as HTMLSelectElement).value), render()) },
              ...state.presets.map((p) => option(p.name, p.builtin ? `${p.name} preset` : `${p.name} (your preset)`, p.name === state.preset)),
              option('custom', 'Custom JSON', state.preset === 'custom'),
            ),
          )
        : null,
    h('span', { class: 'flex-1' }),
    h('span', { class: 'text-[11px]' }, mac ? '⌘ Return' : 'Ctrl + Enter'),
    h('button', { id: 'run-button', class: btnPrimary, disabled: state.running || !state.input.trim(), onClick: () => void run() }, icon(state.running ? ICON.spinner : ICON.play, 'size-3.5'), 'Run'),
  )
  const textarea = h('textarea', {
    class: 'block min-h-0 w-full flex-1 resize-none bg-transparent p-4 text-[13px] leading-relaxed text-fg placeholder:text-muted focus:outline-none',
    placeholder: 'Paste a message, an email or a JSON object…',
    'aria-label': 'State',
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
  const input = h(
    'div',
    { class: 'flex min-w-0 flex-1 flex-col border-r border-line' },
    h('div', { class: 'flex h-8 shrink-0 items-center justify-between border-b border-line px-4' }, h('span', { class: label }, 'State'), h('button', { class: 'text-[11px] text-muted hover:text-fg', onClick: () => ((state.input = EXAMPLE), render()) }, 'Use an example')),
    textarea,
  )
  if (own === null && state.preset === 'custom') {
    const q = h('textarea', {
      class: `block h-40 w-full shrink-0 resize-none bg-transparent p-4 ${mono} leading-relaxed text-fg focus:outline-none`,
      spellcheck: 'false',
      'aria-label': 'Questions as JSON',
      onInput: (e) => (state.customQuestions = (e.target as HTMLTextAreaElement).value),
    }) as HTMLTextAreaElement
    q.value = state.customQuestions
    input.append(h('div', { class: 'flex h-8 shrink-0 items-center border-y border-line px-4' }, h('span', { class: label }, 'Questions')), q)
  }
  const out = h(
    'div',
    { id: 'main-pane', class: 'flex min-w-0 flex-1 flex-col overflow-y-auto' },
    h('div', { class: 'flex h-8 shrink-0 items-center border-b border-line px-4' }, h('span', { class: label }, 'Answers')),
    state.runError
      ? h('p', { class: 'p-4 text-xs text-bad' }, state.runError)
      : state.result
        ? results(state.result)
        : empty(state.running ? 'Running…' : 'The answers appear here: one per question, with how sure the model is.'),
  )
  return h('div', { class: 'flex min-h-0 flex-1 flex-col' }, toolbar, h('div', { class: 'flex min-h-0 flex-1' }, input, out))
}

/** One row per answer: the question, the answer, a meter of ten segments and its probability. */
function results(r: DecideResponse): HTMLElement {
  // qwen3guard's `category` is the harm it would name if it judged the text unsafe, whatever the
  // text (docs/families/qwen3guard.md), so it means little while `unsafe` is low.
  const guard = r.answers.unsafe?.type === 'noul' && !!state.builtin.get(state.runModel)?.includes('category') && 'category' in r.answers
  const quiet = guard && (r.answers.unsafe?.noul ?? 0) < 0.5
  const rows = Object.entries(r.answers).map(([q, a]) => {
    const [text, note, p] = describe(a)
    const lit = Math.round(p * 10)
    return h(
      'li',
      { class: `border-b border-line px-4 py-2.5${quiet && q === 'category' ? ' opacity-50' : ''}` },
      h('p', { class: `${mono} text-muted` }, q),
      h(
        'div',
        { class: 'mt-1 flex items-center gap-3' },
        h('span', { class: 'w-40 shrink-0 truncate text-[13px] font-medium text-fg', title: [text, note].filter(Boolean).join(': ') }, text, note ? h('span', { class: 'ml-1.5 text-xs font-normal text-muted' }, note) : null),
        h('span', { class: 'meter min-w-0 flex-1', 'aria-hidden': 'true' }, ...Array.from({ length: 10 }, (_, i) => h('i', { class: i < lit ? 'on' : undefined }))),
        h('span', { class: 'w-9 text-right text-xs text-fg tabular-nums' }, p.toFixed(2)),
      ),
    )
  })
  return h(
    'div',
    {},
    h('ul', { role: 'list' }, ...rows),
    h('p', { class: 'px-4 py-2.5 text-[11px] text-muted' }, 'Answered by ', h('span', { class: `${mono} text-body` }, r.routing?.model ?? r.model), ' in ', h('span', { class: 'text-fg' }, `${ms(r.total_duration / 1e6)} ms`)),
    guard ? h('p', { class: 'px-4 pb-3 text-[11px] text-muted' }, 'category is the kind of harm the model would name if it judged the text unsafe. It only means something when unsafe is high.') : null,
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

// --- view: server ---------------------------------------------------------------------------------

function server(): HTMLElement {
  const s = state.status
  if (!s?.running)
    return empty(
      s === null ? 'Checking the server…' : 'The server is stopped. Start it to download and run models.',
      h('button', { class: btnPrimary, disabled: state.statusBusy || s === null, onClick: toggleServer }, 'Start'),
    )
  const a = state.activity
  const sp = a ? speed(a) : null
  const median = sp?.median ?? null
  const tile = (value: string, what: string, note: string) =>
    h('div', { class: 'px-4 py-3' }, h('p', { class: 'text-xl font-semibold tracking-tight text-fg tabular-nums' }, value), h('p', { class: 'text-xs text-body' }, what), h('p', { class: 'text-[11px] text-muted' }, note))
  const endpoints: [string, string, string][] = [
    ['POST', '/api/decide', 'Answers questions about a text or JSON, with a preset or your own questions'],
    ['POST', '/v1/decisions', 'TypeSafe-compatible: its SDKs work unchanged (also /v1/systemone)'],
    ['GET', '/v1/models', 'The models on this computer'],
  ]
  const model = allInstalled()[0] ?? 'largha'
  const example = `curl ${s.url}/api/decide -d '${JSON.stringify({ model: model.replace(/:latest$/, ''), preset: 'triage', state: 'I was charged twice this month.' })}'`
  const row = (...children: Child[]) => h('div', { class: 'flex min-h-10 items-center gap-3 border-t border-line px-3 py-1.5 first:border-t-0' }, ...children)
  return h(
    'div',
    { id: 'main-pane', class: 'min-w-0 flex-1 overflow-y-auto' },
    h(
      'div',
      { class: 'mx-auto max-w-3xl p-6' },
      h(
        'div',
        { class: `${panel} grid grid-cols-4 divide-x divide-line` },
        tile(a ? a.total.toLocaleString() : '–', 'Requests', 'since the server started'),
        tile(sp ? sp.perSecond.toFixed(sp.perSecond < 10 ? 1 : 0) : '–', 'Per second', 'over the last minute'),
        tile(median === null ? '–' : `${ms(median)} ms`, 'Response time', sp?.samples ? `median of the last ${sp.samples}` : 'no requests yet'),
        tile(median === null ? '–' : `${Math.floor(1000 / median).toLocaleString()}/s`, 'Capacity', 'one request at a time'),
      ),
      h('div', { class: 'mt-6 flex items-baseline justify-between' }, h('p', { class: label }, 'API'), h('button', { class: 'inline-flex items-center gap-1 text-[11px] text-muted hover:text-fg', onClick: () => void backend.openLink('https://ollaya.dev/docs/api') }, 'Reference', icon(ICON.external, 'size-3'))),
      h(
        'div',
        { class: `${panel} mt-2 overflow-hidden text-xs` },
        row(h('span', { class: 'w-11 text-muted' }, 'Address'), h('span', { class: `${mono} flex-1 text-fg` }, s.url), copyButton(s.url, 'Copy the address')),
        ...endpoints.map(([method, path, what]) =>
          row(h('span', { class: `w-11 ${mono} text-[11px] font-semibold text-muted` }, method), h('span', { class: `w-32 shrink-0 ${mono} text-fg` }, path), h('span', { class: 'min-w-0 flex-1 truncate text-muted', title: what }, what), copyButton(s.url + path, `Copy ${path}`)),
        ),
        h('div', { class: 'flex items-start gap-2 border-t border-line bg-code px-3 py-2.5' }, h('code', { class: `min-w-0 flex-1 ${mono} leading-relaxed wrap-anywhere text-body` }, example), copyButton(example, 'Copy the example')),
      ),
      h('p', { class: `${label} mt-6` }, 'Recent requests'),
      recent(),
    ),
  )
}

function recent(): HTMLElement {
  const a = state.activity
  if (!a) return h('p', { class: 'mt-2 text-xs text-muted' }, state.activityError ? 'This server does not list its requests. Update Ollaya to 0.17.0 or newer.' : 'Loading…')
  if (!a.requests.length)
    return h('p', { class: 'mt-2 text-xs text-muted' }, 'No requests yet. ', h('button', { class: 'text-fg underline underline-offset-4', onClick: () => go('playground') }, 'Try a model'), ', or call the API from your code.')
  const cell = 'px-3 py-1.5'
  return h(
    'div',
    { class: `${panel} mt-2 overflow-hidden` },
    h(
      'table',
      { class: 'w-full text-xs' },
      h('thead', { class: 'text-left text-[11px] text-muted' }, h('tr', {}, ...['Time', 'Model', 'Endpoint', 'Questions', 'Took'].map((t, i) => h('th', { class: `${cell} font-medium ${i > 2 ? 'text-right' : ''}` }, t)))),
      h(
        'tbody',
        {},
        ...a.requests.slice(0, 20).map((r) =>
          h(
            'tr',
            { class: 'border-t border-line' },
            h('td', { class: `${cell} text-muted tabular-nums` }, new Date(r.at).toLocaleTimeString()),
            h('td', { class: `${cell} ${mono} text-fg` }, r.model),
            h('td', { class: `${cell} ${mono} text-muted` }, r.endpoint),
            h('td', { class: `${cell} text-right tabular-nums ${r.error ? 'text-bad' : 'text-body'}` }, r.error ? r.error.toLowerCase().replaceAll('_', ' ') : String(r.questions)),
            h('td', { class: `${cell} text-right text-body tabular-nums` }, `${ms(r.total_duration / 1e6)} ms`),
          ),
        ),
      ),
    ),
  )
}

// --- render loop ----------------------------------------------------------------------------------

const root = document.getElementById('app')!

function render() {
  // Keep the focused field's caret and the panes' scroll across redraws.
  const active = document.activeElement as HTMLInputElement | HTMLTextAreaElement | null
  const fieldName = active && ['TEXTAREA', 'INPUT'].includes(active.tagName) ? active.getAttribute('aria-label') : null
  const caret = fieldName ? ([active!.selectionStart, active!.selectionEnd] as const) : null
  const panes = ['main-pane', 'list-pane']
  const scrolled = panes.map((id) => document.getElementById(id)?.scrollTop ?? 0)
  const page = state.page === 'discover' ? discover() : state.page === 'mine' ? mine() : state.page === 'playground' ? playground() : server()
  root.replaceChildren(h('div', { class: 'flex min-h-0 flex-1' }, rail(), h('main', { class: 'flex min-w-0 flex-1 flex-col' }, page)), statusBar())
  panes.forEach((id, i) => document.getElementById(id)?.scrollTo({ top: scrolled[i] }))
  if (fieldName && caret) {
    const again = root.querySelector<HTMLInputElement | HTMLTextAreaElement>(`[aria-label="${fieldName}"]`)
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
  // With models on this computer, start where they run; a first run starts where they are found.
  if (state.installed.size) state.page = 'playground'
  // Custom presets live on the server, which may only have started just now.
  state.presets = await backend.presets().catch(() => state.presets)
  state.update = await backend.updateAvailable().catch(() => null)
  render()
  setInterval(() => void refresh(), 2000)
  // Preview only (npm run preview): #result shows a finished run, #pull a download in progress.
  if (__MOCK__ && location.hash === '#result') {
    tryOut('laya:latest')
    state.input = EXAMPLE
    await run()
  }
  if (__MOCK__ && location.hash === '#pull') (state.selected = 'decider'), go('discover'), void pull('decider:latest')
}

declare const __MOCK__: boolean

void boot()
