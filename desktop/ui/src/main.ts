// Ollaya desktop: the server's status and switch, the model library with downloads, and a panel to
// run a model on a text or JSON state. Plain DOM, rendered from one state object.
import { backend, type Answer, type DecideResponse, type LibraryModel, type Preset, type Status } from './backend'

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

const OWL = `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 5.5 10.5 9.5Q16 7.5 21.5 9.5L26 5.5V19Q26 27.5 16 27.5T6 19Z"/><circle cx="11.5" cy="15.5" r="3.25"/><circle cx="20.5" cy="15.5" r="3.25"/><circle cx="11.5" cy="15.5" r="1.1" fill="currentColor" stroke="none"/><circle cx="20.5" cy="15.5" r="1.1" fill="currentColor" stroke="none"/><path d="M14.75 21 16 23l1.25-2"/></svg>`
const ICON = {
  download: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 3v10m0 0 4-4m-4 4-4-4M4 16h12"/></svg>`,
  check: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m5 10.5 3.2 3.2L15 7"/></svg>`,
  x: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><path d="m6 6 8 8M14 6l-8 8"/></svg>`,
  spinner: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true" class="spin"><circle cx="10" cy="10" r="7" stroke="currentColor" stroke-opacity=".25" stroke-width="2"/><path d="M17 10a7 7 0 0 0-7-7" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
}
function icon(svg: string, cls: string): HTMLElement {
  const s = h('span', { class: `inline-flex shrink-0 ${cls}` })
  s.innerHTML = svg // constant markup, never data
  return s
}

// --- state ----------------------------------------------------------------------------------------

interface Pulling {
  completed: number
  total: number
  status: string
}

const state = {
  status: null as Status | null,
  statusBusy: false,
  statusError: '',
  library: [] as LibraryModel[],
  libraryError: '',
  installed: new Set<string>(),
  selected: '',
  pulling: new Map<string, Pulling>(),
  pullError: '',
  presets: [] as Preset[],
  runModel: '',
  preset: 'triage',
  input: '',
  customQuestions: '',
  running: false,
  result: null as DecideResponse | null,
  runError: '',
}

/**
 * The versions to offer: `<model>:latest` first (what `ollaya run <model>` uses, which the
 * library index doesn't always list), then the main tags. Precision variants (-fp16, -fp32) are
 * for pinning, not for picking.
 */
const mainTags = (m: LibraryModel) => {
  const tags = m.tags.map((t) => t.name).filter((n) => !/-fp(16|32)$/.test(n))
  const latest = `${m.name}:latest`
  return [latest, ...tags.filter((t) => t !== latest)]
}
const tagSummary = (m: LibraryModel, tag: string) =>
  m.tags.find((t) => t.name === tag)?.summary || (tag.endsWith(':latest') ? `The default version: ollaya run ${m.name}` : '')
const installedTags = (m: LibraryModel) => mainTags(m).filter((t) => state.installed.has(t))
const selectedModel = () => state.library.find((m) => m.name === state.selected)

// --- actions --------------------------------------------------------------------------------------

async function refreshInstalled() {
  if (!state.status?.running) {
    state.installed = new Set()
    return
  }
  try {
    state.installed = new Set((await backend.installed()).map((m) => m.name))
  } catch {
    state.installed = new Set()
  }
}

/** Poll the server; redraw only when something changed, so an open menu or a caret survives. */
async function refreshStatus() {
  const before = JSON.stringify([state.status, [...state.installed]])
  state.status = await backend.status()
  await refreshInstalled()
  if (JSON.stringify([state.status, [...state.installed]]) !== before) render()
}

async function toggleServer() {
  state.statusBusy = true
  state.statusError = ''
  render()
  try {
    state.status = state.status?.running ? await backend.stopServer() : await backend.startServer()
    await refreshInstalled()
  } catch (e) {
    state.statusError = String(e)
  }
  state.statusBusy = false
  render()
}

async function pull(tag: string) {
  state.pullError = ''
  state.pulling.set(tag, { completed: 0, total: 0, status: 'starting' })
  render()
  try {
    await backend.pull(tag)
  } catch (e) {
    state.pullError = String(e)
  }
  state.pulling.delete(tag)
  await refreshInstalled()
  render()
}

async function remove(tag: string) {
  try {
    await backend.remove(tag)
  } catch (e) {
    state.pullError = String(e)
  }
  await refreshInstalled()
  if (state.runModel === tag) state.runModel = ''
  render()
}

async function run() {
  const model = state.runModel || installedTags(selectedModel()!)[0]
  if (!model || !state.input.trim()) return
  state.running = true
  state.runError = ''
  render()
  try {
    state.result =
      state.preset === 'custom'
        ? await backend.decide(model, state.input, null, state.customQuestions)
        : await backend.decide(model, state.input, state.preset, null)
  } catch (e) {
    state.result = null
    state.runError = String(e)
  }
  state.running = false
  render()
  // Bring the answers into view inside the main pane (they land below the fold on a small window).
  // Not scrollIntoView: it would also scroll the page and push the header out of sight.
  const pane = document.getElementById('main-pane')
  const answers = document.getElementById('answers')
  if (pane && answers) {
    const bottom = answers.offsetTop + answers.offsetHeight + 24 - pane.clientHeight
    if (bottom > pane.scrollTop) pane.scrollTop = bottom
  }
}

function select(name: string) {
  if (state.selected !== name) {
    document.getElementById('main-pane')?.scrollTo({ top: 0 })
    state.selected = name
    state.runModel = ''
    state.result = null
    state.runError = ''
  }
  render()
}

// --- view -----------------------------------------------------------------------------------------

const btnPrimary =
  'inline-flex h-8 items-center justify-center gap-1.5 rounded-full bg-btn px-4 text-[13px] font-medium text-btn-fg hover:bg-btn-hover disabled:opacity-50'
const btnSecondary =
  'inline-flex h-8 items-center justify-center gap-1.5 rounded-full border border-line-strong px-4 text-[13px] font-medium text-fg hover:bg-fill disabled:opacity-50'

function header(): HTMLElement {
  const s = state.status
  const running = !!s?.running
  return h(
    'header',
    { class: 'flex h-14 shrink-0 items-center justify-between border-b border-line px-5' },
    h(
      'div',
      { class: 'flex items-center gap-2 text-fg' },
      icon(OWL, '-ml-[4px] size-6'),
      h('span', { class: 'text-[17px] leading-6 font-medium tracking-tight' }, 'ollaya'),
      h('span', { class: 'rounded-full border border-line-strong px-[7px] text-[11px] leading-[17px] font-medium text-muted' }, 'Beta'),
    ),
    h(
      'div',
      { class: 'flex items-center gap-3' },
      state.statusError ? h('span', { class: 'max-w-80 truncate text-xs text-bad', title: state.statusError }, state.statusError) : null,
      h(
        'span',
        { class: 'flex items-center gap-2 text-[13px] text-muted' },
        h('span', { class: `size-2 rounded-full ${running ? 'bg-ok' : 'bg-faint'}` }),
        s === null ? 'Checking…' : running ? `Running · ${s.version ?? ''}` : 'Stopped',
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
  return h(
    'nav',
    { class: 'flex w-60 shrink-0 flex-col border-r border-line bg-subtle', 'aria-label': 'Models' },
    h('p', { class: 'px-5 pt-5 pb-2 text-[11px] font-semibold tracking-wide text-muted uppercase' }, 'Models'),
    state.libraryError
      ? h('p', { class: 'px-5 text-xs text-bad' }, state.libraryError)
      : h(
          'ul',
          { class: 'space-y-0.5 px-2.5' },
          ...state.library.map((m) => {
            const tags = mainTags(m)
            const pulling = tags.find((t) => state.pulling.has(t))
            const have = installedTags(m).length > 0
            const on = m.name === state.selected
            return h(
              'li',
              {},
              h(
                'button',
                {
                  class: `flex h-9 w-full items-center justify-between rounded-lg px-2.5 text-left text-[13px] ${on ? 'bg-fill-strong font-medium text-fg' : 'text-body hover:bg-fill'}`,
                  'aria-current': on ? 'true' : undefined,
                  onClick: () => select(m.name),
                },
                h('span', { class: 'font-mono' }, m.name),
                pulling
                  ? h('span', { class: 'text-xs text-muted tabular-nums' }, percent(state.pulling.get(pulling)!))
                  : have
                    ? icon(ICON.check, 'size-4 text-ok')
                    : icon(ICON.download, 'size-4 text-faint'),
              ),
            )
          }),
        ),
  )
}

function percent(p: { completed: number; total: number }): string {
  return p.total > 0 ? `${Math.floor((p.completed / p.total) * 100)}%` : '…'
}

function main(): HTMLElement {
  const m = selectedModel()
  const body = h('div', { class: 'mx-auto w-full max-w-3xl px-8 py-8' })
  if (!m) {
    body.append(h('p', { class: 'pt-24 text-center text-sm text-muted' }, state.library.length ? 'Choose a model.' : 'Loading the model library…'))
    return h('main', { id: 'main-pane', class: 'relative min-w-0 flex-1 overflow-y-auto' }, body)
  }
  const running = !!state.status?.running
  const tags = mainTags(m)
  body.append(
    h('h1', { class: 'font-mono text-2xl font-semibold tracking-tight text-fg' }, m.name),
    h('p', { class: 'mt-2 max-w-2xl text-sm leading-relaxed text-muted' }, m.description),
    h(
      'div',
      { class: 'mt-3 flex flex-wrap gap-1.5' },
      ...m.caps.map((c) => h('span', { class: 'rounded-md bg-fill px-2 py-0.5 text-xs text-body' }, c)),
    ),
    h('h2', { class: 'mt-8 text-[13px] font-semibold text-fg' }, 'Versions'),
    h(
      'div',
      { class: 'mt-3 overflow-hidden rounded-xl border border-line' },
      ...tags.map((t, i) => versionRow(m, t, i, running)),
    ),
  )
  if (state.pullError) body.append(h('p', { class: 'mt-3 text-xs text-bad' }, state.pullError))
  if (!running) {
    body.append(h('p', { class: 'mt-8 text-sm text-muted' }, 'Start the server to download and run models.'))
  } else if (installedTags(m).length) {
    body.append(runPanel(m))
  }
  return h('main', { id: 'main-pane', class: 'relative min-w-0 flex-1 overflow-y-auto' }, body)
}

function versionRow(m: LibraryModel, tag: string, i: number, running: boolean): HTMLElement {
  const summary = tagSummary(m, tag)
  const have = state.installed.has(tag)
  const p = state.pulling.get(tag)
  let action: HTMLElement
  if (p) {
    action = h(
      'span',
      { class: 'flex w-40 items-center gap-2.5' },
      h(
        'span',
        { class: 'h-1.5 flex-1 overflow-hidden rounded-full bg-fill-strong' },
        h('span', { class: 'bar block h-full rounded-full transition-[width]', style: `width:${p.total ? (p.completed / p.total) * 100 : 2}%` }),
      ),
      h('span', { class: 'w-9 text-right text-xs text-muted tabular-nums' }, percent(p)),
    )
  } else if (have) {
    action = h(
      'span',
      { class: 'flex items-center gap-1' },
      h('span', { class: 'flex items-center gap-1 text-xs text-muted' }, icon(ICON.check, 'size-4 text-ok'), 'Installed'),
      h(
        'button',
        { class: 'ml-2 inline-flex size-7 items-center justify-center rounded-full text-faint hover:bg-fill hover:text-fg', title: `Remove ${tag}`, 'aria-label': `Remove ${tag}`, onClick: () => void remove(tag) },
        icon(ICON.x, 'size-4'),
      ),
    )
  } else {
    action = h('button', { class: btnSecondary, disabled: !running, onClick: () => void pull(tag) }, icon(ICON.download, 'size-4'), 'Download')
  }
  return h(
    'div',
    { class: `flex min-h-14 items-center justify-between gap-4 px-4 py-2.5 ${i ? 'border-t border-line' : ''}` },
    h(
      'div',
      { class: 'min-w-0' },
      h('p', { class: 'font-mono text-[13px] text-fg' }, tag),
      summary ? h('p', { class: 'mt-0.5 truncate text-xs text-muted' }, summary) : null,
    ),
    action,
  )
}

function runPanel(m: LibraryModel): HTMLElement {
  const tags = installedTags(m)
  if (!tags.includes(state.runModel)) state.runModel = tags[0] ?? ''
  const selectCls = 'h-8 rounded-lg border border-line-strong bg-canvas px-2.5 text-[13px] text-fg'
  const textarea = h('textarea', {
    class: 'mt-3 block h-28 w-full resize-y rounded-xl border border-line-strong bg-canvas px-3.5 py-3 text-sm leading-relaxed text-fg placeholder:text-faint focus:border-fg focus:outline-none',
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
  const panel = h(
    'section',
    { class: 'mt-10', 'aria-labelledby': 'run-title' },
    h('h2', { id: 'run-title', class: 'text-[13px] font-semibold text-fg' }, 'Run'),
    h(
      'div',
      { class: 'mt-3 flex flex-wrap items-center gap-2' },
      labelled('Model', h('select', { class: selectCls, onChange: (e) => ((state.runModel = (e.target as HTMLSelectElement).value), render()) }, ...tags.map((t) => option(t, t, t === state.runModel)))),
      labelled(
        'Questions',
        h(
          'select',
          { class: selectCls, onChange: (e) => ((state.preset = (e.target as HTMLSelectElement).value), render()) },
          ...state.presets.map((p) => option(p.name, `${p.name} preset`, p.name === state.preset)),
          option('custom', 'Custom JSON', state.preset === 'custom'),
        ),
      ),
    ),
    textarea,
  )
  if (state.preset === 'custom') {
    const q = h('textarea', {
      class: 'mt-2 block h-40 w-full resize-y rounded-xl border border-line-strong bg-code px-3.5 py-3 font-mono text-xs leading-relaxed text-fg focus:border-fg focus:outline-none',
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
      { class: 'mt-3 flex items-center justify-end gap-3' },
      h('span', { class: 'text-xs text-faint' }, mac ? '⌘ Return' : 'Ctrl + Enter'),
      h('button', { id: 'run-button', class: btnPrimary, disabled: state.running || !state.input.trim(), onClick: () => void run() }, state.running ? icon(ICON.spinner, 'size-3.5') : null, 'Run'),
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

/** One row per answer, as the CLI prints them: the answer, a bar and its probability. */
function results(r: DecideResponse): HTMLElement {
  const rows = Object.entries(r.answers).map(([q, a]) => {
    const [text, note, p] = describe(a)
    return h(
      'div',
      { class: 'grid grid-cols-[minmax(0,11rem)_minmax(0,1fr)_7rem_2.75rem] items-center gap-x-4 py-2 border-t border-line first:border-t-0' },
      h('span', { class: 'truncate font-mono text-xs text-muted' }, q),
      h('span', { class: 'truncate text-sm font-medium text-fg' }, text, note ? h('span', { class: 'ml-2 font-normal text-muted' }, note) : null),
      h('span', { class: 'h-1.5 overflow-hidden rounded-full bg-fill-strong' }, h('span', { class: 'bar block h-full rounded-full', style: `width:${Math.round(p * 100)}%` })),
      h('span', { class: 'text-right font-mono text-xs text-body tabular-nums' }, p.toFixed(2)),
    )
  })
  const ms = r.total_duration / 1e6
  return h(
    'div',
    { id: 'answers', class: 'mt-6 pb-2' },
    h('div', { class: 'rounded-xl border border-line px-4 py-1' }, ...rows),
    h('p', { class: 'mt-3 text-xs text-muted' }, `Answered by ${r.routing?.model ?? r.model} in ${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`),
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
  // Keep the focused textarea's caret across redraws.
  const active = document.activeElement as HTMLTextAreaElement | null
  const focusedLabel = active?.tagName === 'TEXTAREA' ? active.getAttribute('aria-label') ?? 'state' : null
  const caret = focusedLabel ? ([active!.selectionStart, active!.selectionEnd] as const) : null
  const scrolled = document.getElementById('main-pane')?.scrollTop ?? 0
  root.replaceChildren(header(), h('div', { class: 'flex min-h-0 flex-1' }, sidebar(), main()))
  document.getElementById('main-pane')?.scrollTo({ top: scrolled })
  if (focusedLabel && caret) {
    const again = [...root.querySelectorAll('textarea')].find((t) => (t.getAttribute('aria-label') ?? 'state') === focusedLabel)
    again?.focus()
    again?.setSelectionRange(caret[0], caret[1])
  }
}

backend.onPullProgress((p) => {
  if (p.done) return
  state.pulling.set(p.model, { completed: p.completed, total: p.total, status: p.status })
  render()
})

async function boot() {
  render()
  const [lib, presets] = await Promise.allSettled([backend.library(), backend.presets()])
  if (lib.status === 'fulfilled') {
    state.library = lib.value
    state.selected = lib.value[0]?.name ?? ''
  } else {
    state.libraryError = `Could not load the model library: ${lib.reason}`
  }
  if (presets.status === 'fulfilled') {
    state.presets = presets.value
    state.customQuestions = JSON.stringify(presets.value.find((p) => p.name === 'triage')?.questions ?? {}, null, 2)
  }
  await refreshStatus()
  render()
  // Like Ollama's app: the server runs while the app is open.
  if (!state.status?.running) await toggleServer()
  setInterval(() => void refreshStatus(), 5000)
  // Preview only (npm run preview): #result shows a finished run, #pull a download in progress.
  if (__MOCK__ && location.hash === '#result') {
    state.input = "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
    await run()
  }
  if (__MOCK__ && location.hash === '#pull') void pull('decider:latest'), select('decider')
}

declare const __MOCK__: boolean

void boot()
