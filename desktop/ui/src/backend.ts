// The app's commands (src-tauri/src/lib.rs, and cloud.rs for Ollaya Cloud), called through Tauri's global API. A preview build
// (`npm run preview`) swaps in a fake backend, so the page can be designed in a browser.

import { CloudError } from './cloud-error'

export { CloudError }

declare const __MOCK__: boolean

export interface Status {
  running: boolean
  version: string | null
  url: string
  /** Where the server this app started runs models (Windows and Linux): "GPU" or "CPU only". */
  device: string | null
}

/** A version of a library model, as ollaya.dev/data/library.json lists it. */
export interface LibraryTag {
  name: string
  summary: string
  /** Download size in bytes (a router's: the models it routes to). */
  size: number
  params: string | null
  languages: string
}

export interface LibraryModel {
  name: string
  title: string
  description: string
  publisher: string | null
  license: string
  caps: string[]
  /** Editorial order, lower first. */
  rank: number
  /** Measured on the benchmark GPU, for `tag`. */
  stats: { tag: string; accuracy?: number; latencyMs?: number } | null
  tags: LibraryTag[]
  /** The model card as HTML, from ollaya.dev. */
  readme: string | null
}

/** One decision request the server answered (`GET /api/activity`). */
export interface Activity {
  at: string
  endpoint: string
  model: string
  questions: number
  /** Nanoseconds. */
  total_duration: number
  /** The error code, when it failed. */
  error?: string
}

export interface ActivityResponse {
  total: number
  /** Newest first. */
  requests: Activity[]
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
  /** False for a preset saved with `ollaya preset create`. */
  builtin: boolean
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

/** Where the Playground's decisions run. */
export type CloudMode = 'local' | 'cloud'

/** What the app tells the page about Ollaya Cloud. The session and the key stay in Rust. */
export interface CloudStatus {
  signed_in: boolean
  email: string | null
  mode: CloudMode
  /** While signing in: the code to compare with the one the browser shows. */
  sign_in: { user_code: string; verification_uri: string } | null
  /** Why the last sign-in failed. */
  error: string | null
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
  /** The names of the models loaded in memory. */
  loaded(): Promise<string[]>
  activity(): Promise<ActivityResponse>
  pull(model: string): Promise<void>
  remove(model: string): Promise<void>
  presets(): Promise<Preset[]>
  /** The questions a model asks by itself, or null when requests must bring them. */
  builtinQuestions(model: string): Promise<Questions | null>
  /** With neither `preset` nor `questions`, the model answers its built-in questions. */
  decide(model: string, state: string, preset: string | null, questions: string | null): Promise<DecideResponse>
  onPullProgress(handler: (p: PullProgress) => void): void
  /** A newer Ollaya release than this app, or null. */
  updateAvailable(): Promise<string | null>
  /** Open the download page in the browser. */
  openDownload(): Promise<void>
  /** Open a web link in the browser. */
  openLink(url: string): Promise<void>
  cloudStatus(): Promise<CloudStatus>
  /** Ask for a code and open the browser on it; the outcome comes through `onCloudStatus`. */
  cloudSignIn(): Promise<CloudStatus>
  /** Open the browser on the code again. */
  cloudOpenSignIn(): Promise<void>
  cloudCancelSignIn(): Promise<CloudStatus>
  cloudSignOut(): Promise<CloudStatus>
  cloudSetMode(mode: CloudMode): Promise<CloudStatus>
  /** The names of the models Ollaya Cloud runs, which differ from local ones. */
  cloudModels(): Promise<string[]>
  /** `decide` on Ollaya Cloud. Fails with a `CloudError`. */
  cloudDecide(model: string, state: string, preset: string | null, questions: string | null): Promise<DecideResponse>
  onCloudStatus(handler: (s: CloudStatus) => void): void
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
    loaded: async () => (await invoke<{ models: { name: string }[] }>('loaded')).models.map((m) => m.name),
    activity: () => invoke('activity'),
    pull: (model) => invoke('pull', { model }),
    remove: (model) => invoke('remove', { model }),
    presets: () => invoke('preset_list'),
    builtinQuestions: (model) => invoke('builtin_questions', { model }),
    decide: (model, state, preset, questions) => invoke('decide', { model, state, preset, questions }),
    onPullProgress: (handler) => {
      void t.event.listen<PullProgress>('pull-progress', (e) => handler(e.payload))
    },
    updateAvailable: () => invoke('update_available'),
    openDownload: () => invoke('open_download'),
    openLink: (url) => invoke('open_link', { url }),
    cloudStatus: () => invoke('cloud_status'),
    cloudSignIn: () => cloud(invoke('cloud_sign_in')),
    cloudOpenSignIn: () => invoke('cloud_open_sign_in'),
    cloudCancelSignIn: () => invoke('cloud_cancel_sign_in'),
    cloudSignOut: () => cloud(invoke('cloud_sign_out')),
    cloudSetMode: (mode) => invoke('cloud_set_mode', { mode }),
    cloudModels: () => cloud(invoke('cloud_models')),
    cloudDecide: (model, state, preset, questions) => cloud(invoke('cloud_decide', { model, state, preset, questions })),
    onCloudStatus: (handler) => {
      void t.event.listen<CloudStatus>('cloud-status', (e) => handler(e.payload))
    },
  }
}

/** The cloud commands fail with `{ code, message }` (cloud.rs `UiError`). */
async function cloud<T>(p: Promise<T>): Promise<T> {
  try {
    return await p
  } catch (e) {
    const err = e as { code?: unknown; message?: unknown }
    throw typeof err?.message === 'string' ? new CloudError(String(err.code ?? ''), err.message) : new CloudError('', String(e))
  }
}

export const backend: Backend = __MOCK__ ? (await import('./mock')).mock() : tauri()
