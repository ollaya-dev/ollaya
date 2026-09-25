// The app's commands (src-tauri/src/lib.rs), called through Tauri's global API. A preview build
// (`npm run preview`) swaps in a fake backend, so the page can be designed in a browser.

declare const __MOCK__: boolean

export interface Status {
  running: boolean
  version: string | null
  url: string
}

export interface LibraryTag {
  name: string
  summary: string
}

export interface LibraryModel {
  name: string
  description: string
  caps: string[]
  tags: LibraryTag[]
}

export interface LocalModel {
  name: string
  size: number
}

export interface PullProgress {
  model: string
  status: string
  completed: number
  total: number
  done: boolean
  error: string | null
}

/** A question schema: question id → its definition. */
export type Questions = Record<string, { type: string }>

export interface Preset {
  name: string
  questions: Questions
}

export interface Answer {
  type: 'choice' | 'score' | 'noul'
  choice?: string
  confidence?: number
  score?: number
  legend?: Record<string, string>
  noul?: number
}

export interface DecideResponse {
  model: string
  answers: Record<string, Answer>
  routing: { model: string } | null
  total_duration: number
}

interface TauriGlobal {
  core: { invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> }
  event: { listen<T>(event: string, handler: (e: { payload: T }) => void): Promise<() => void> }
}

declare global {
  interface Window {
    __TAURI__?: TauriGlobal
  }
}

export interface Backend {
  status(): Promise<Status>
  startServer(): Promise<Status>
  stopServer(): Promise<Status>
  library(): Promise<LibraryModel[]>
  installed(): Promise<LocalModel[]>
  pull(model: string): Promise<void>
  remove(model: string): Promise<void>
  presets(): Promise<Preset[]>
  /** The questions a model asks by itself, or null when requests must bring them. */
  builtinQuestions(model: string): Promise<Questions | null>
  /** With neither `preset` nor `questions`, the model answers its built-in questions. */
  decide(model: string, state: string, preset: string | null, questions: string | null): Promise<DecideResponse>
  onPullProgress(handler: (p: PullProgress) => void): void
}

function tauri(): Backend {
  const t = window.__TAURI__
  if (!t) throw new Error('not running inside the Ollaya app')
  const invoke = t.core.invoke
  return {
    status: () => invoke('status'),
    startServer: () => invoke('start_server'),
    stopServer: () => invoke('stop_server'),
    library: () => invoke('library'),
    installed: async () => (await invoke<{ models: LocalModel[] }>('installed')).models,
    pull: (model) => invoke('pull', { model }),
    remove: (model) => invoke('remove', { model }),
    presets: () => invoke('preset_list'),
    builtinQuestions: (model) => invoke('builtin_questions', { model }),
    decide: (model, state, preset, questions) => invoke('decide', { model, state, preset, questions }),
    onPullProgress: (handler) => {
      void t.event.listen<PullProgress>('pull-progress', (e) => handler(e.payload))
    },
  }
}

export const backend: Backend = __MOCK__ ? (await import('./mock')).mock() : tauri()
