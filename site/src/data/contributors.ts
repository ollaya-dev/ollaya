/**
 * People who brought a model family to Ollaya: who ported it (a pull request) or asked for it (an issue), by model
 * name. The pull request that adds a model adds its line here. The authors come from the catalog.
 */
export interface BroughtBy {
  github: string
  /** Issue or pull request numbers on GitHub. */
  refs: number[]
  what: string
}

export const broughtBy: Record<string, BroughtBy[]> = {
  arbiter: [{ github: 'codemanhitesh', refs: [49], what: 'ported it' }],
  clm: [{ github: 'K0IN', refs: [13], what: 'asked for it' }],
  credence: [{ github: 'Txoka', refs: [66, 68], what: 'ported it' }],
  cygnet: [{ github: 'Xain3', refs: [39], what: 'asked for it' }],
  d1: [{ github: 'Txoka', refs: [72], what: 'ported it' }],
  decider: [{ github: 'K0IN', refs: [14], what: 'asked for the vision tag' }],
  decima: [{ github: 'amyrmahdy', refs: [54, 70], what: 'asked for it, as its author' }],
  jeb: [{ github: 'webdevtodayjason', refs: [18], what: 'asked for it' }],
  snap: [{ github: 'emnlmn', refs: [46], what: 'asked for it' }],
  winnow: [{ github: 'Txoka', refs: [52, 59], what: 'ported the vision tags' }],
}

/** shortcut: authors the catalog has no publisher for yet; move each into catalog.ts's overlay, then drop it here. */
export const authorFallback: Record<string, { name: string; url: string }> = {
  d1: { name: 'Liquid AI', url: 'https://huggingface.co/LiquidAI' },
}
