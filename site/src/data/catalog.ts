/**
 * The model catalog — single source of truth for the search page, the search index
 * (/search.json, used by the navbar typeahead), model pages, tag tables, tag pages and the sitemap.
 *
 * Rules for editing:
 *  - Only list models that Ollaya will actually serve. Upcoming ideas go in `comingNext`.
 *  - Never add pull/download counts or other usage numbers we do not have.
 *  - Sizes are approximate until artifacts are published; digests are not shown yet.
 *  - Readme prose lives in content/library/<model>.md (rendered at build time).
 */

export type Capability = 'multilingual' | 'router' | 'guardrails' | 'fine-tuned'

export interface CapabilityFilter {
  id: Capability
  label: string
}

/** Filter chips on /search, in display order. */
export const capabilityFilters: CapabilityFilter[] = [
  { id: 'multilingual', label: 'multilingual' },
  { id: 'router', label: 'router' },
  { id: 'guardrails', label: 'guardrails' },
  { id: 'fine-tuned', label: 'fine-tuned' },
]

export type Precision = 'auto' | 'fp16' | 'fp32'

/** A pullable artifact layer shown on a tag page ("Details"). */
export interface Layer {
  kind: 'model' | 'tokenizer' | 'decision' | 'calibration' | 'license' | 'router'
  /** Short monospace preview of the layer contents. */
  preview: string
  /** Approximate size, human readable. Omitted when unknown. */
  size?: string
}

export interface Tag {
  /** Tag name without the model, e.g. "en" or "en-fp32". */
  name: string
  kind: 'router' | 'model'
  /** Base tag for precision variants ("en" for "en-fp16"). */
  variantOf?: string
  precision: Precision | null
  /** Tags the router may resolve to (router tags only). */
  routesTo?: string[]
  backbone?: string
  /** Hugging Face id of the encoder backbone. */
  encoder?: string
  params?: string
  /** Context window in tokens (a string for routers, which depend on the resolved model). */
  context: string
  languages: string
  /** Approximate download size in GB per precision. */
  sizeGB?: { fp16: number; fp32: number }
  /** One-line summary for tables and lists. */
  summary: string
  /** Shown in the model page "Models" table (bare tags only). */
  featured: boolean
  capabilities: Capability[]
}

export interface Model {
  name: string
  title: string
  description: string
  publisher: { name: string; url: string }
  source: string
  license: string
  capabilities: Capability[]
  /** Parameter-size badges, lower case like "421m". */
  sizes: string[]
  /** Date this catalog entry was last updated (YYYY-MM-DD). */
  updated: string
  /** Curated order for "Popular" (we have no pull counts, so this is editorial). */
  rank: number
  /** Extra search terms. */
  keywords: string[]
  tags: Tag[]
  /** Example state + preset used in usage snippets. */
  example: { state: string }
}

const LAYA_CAL_PREVIEW = '{"temperature_by_options": {"choice:2": …, "choice:3-5": …, "score:3-5": …, "noul:2": …, …}}'

function precisionVariants(base: Tag): Tag[] {
  return (['fp16', 'fp32'] as const).map((p) => ({
    ...base,
    name: `${base.name}-${p}`,
    variantOf: base.name,
    precision: p,
    featured: false,
  }))
}

const layaEn: Tag = {
  name: 'en',
  kind: 'model',
  precision: 'auto',
  backbone: 'ModernBERT-large',
  encoder: 'answerdotai/ModernBERT-large',
  params: '421M',
  context: '512',
  languages: 'English',
  sizeGB: { fp16: 0.8, fp32: 1.7 },
  summary: 'English. Best for guardrails and email triage.',
  featured: true,
  capabilities: ['guardrails'],
}

const layaMultilingual: Tag = {
  name: 'multilingual',
  kind: 'model',
  precision: 'auto',
  backbone: 'mmBERT-base',
  encoder: 'jhu-clsp/mmBERT-base',
  params: '322M',
  context: '1024',
  languages: '100+ languages',
  sizeGB: { fp16: 0.65, fp32: 1.3 },
  summary: '100+ languages, 1024-token context; up to ~2.2× faster on batched calls.',
  featured: true,
  capabilities: ['multilingual'],
}

const layaTyped: Tag = {
  name: 'typed-decisions',
  kind: 'model',
  precision: 'auto',
  backbone: 'ModernBERT-large',
  encoder: 'answerdotai/ModernBERT-large',
  params: '421M',
  context: '1024',
  languages: 'English',
  // Same backbone as laya:en, so roughly the same size.
  sizeGB: { fp16: 0.8, fp32: 1.7 },
  summary: 'Fine-tuned on typed-decisions workflows: 0.766 accuracy vs 0.727 published for Jev 1.13.',
  featured: true,
  capabilities: ['fine-tuned'],
}

const layaLatest: Tag = {
  name: 'latest',
  kind: 'router',
  precision: null,
  routesTo: ['en', 'multilingual'],
  context: '512 / 1024',
  languages: 'Auto-detected',
  summary: 'Router: picks laya:en or laya:multilingual from the detected script and language.',
  featured: true,
  capabilities: ['router', 'multilingual'],
}

export const catalog: Model[] = [
  {
    name: 'laya',
    title: 'Laya',
    description:
      'Open decision models from Convai Innovations. Typed, calibrated answers to choice, score and yes/no questions in a single forward pass, in English and 100+ languages.',
    publisher: { name: 'Convai Innovations', url: 'https://huggingface.co/convaiinnovations' },
    source: 'https://huggingface.co/convaiinnovations/laya',
    license: 'Apache-2.0',
    capabilities: ['multilingual', 'router', 'guardrails', 'fine-tuned'],
    sizes: ['322m', '421m'],
    updated: '2026-09-23',
    rank: 1,
    keywords: [
      'decision',
      'classification',
      'classifier',
      'triage',
      'moderation',
      'guardrail',
      'routing',
      'modernbert',
      'mmbert',
      'typesafe',
      'jev',
      'system one',
      'convai',
    ],
    tags: [
      layaLatest,
      layaEn,
      ...precisionVariants(layaEn),
      layaMultilingual,
      ...precisionVariants(layaMultilingual),
      layaTyped,
      ...precisionVariants(layaTyped),
    ],
    example: { state: 'I was charged twice for my subscription this month and want a refund.' },
  },
]

/** Not pullable — shown as "Coming next" on /search and the home page. */
export const comingNext: string[] = [
  'von',
  'GLiClass',
  'NLI zero-shot classifiers',
  'GGUF LLM-based decision models via llama.cpp',
]

// ---------------------------------------------------------------------------------------------
// Lookups & helpers

export function getModel(name: string): Model | undefined {
  return catalog.find((m) => m.name === name)
}

export function getTag(model: Model, tag: string): Tag | undefined {
  return model.tags.find((t) => t.name === tag)
}

export function featuredTags(model: Model): Tag[] {
  return model.tags.filter((t) => t.featured)
}

export const fullName = (model: Model, tag: Tag): string => `${model.name}:${tag.name}`

const num = (n: number) => (n < 1 ? n.toFixed(2).replace(/0$/, '') : n.toFixed(1))
const gb = (n: number) => `~${num(n)} GB`

/** Approximate size label for a tag ("~0.8 GB", "~0.8 / 1.7 GB" for auto precision, "—" for routers). */
export function sizeLabel(tag: Tag): string {
  if (!tag.sizeGB) return '—'
  if (tag.precision === 'fp16') return gb(tag.sizeGB.fp16)
  if (tag.precision === 'fp32') return gb(tag.sizeGB.fp32)
  return `~${num(tag.sizeGB.fp16)} / ${num(tag.sizeGB.fp32)} GB`
}

export function precisionLabel(tag: Tag): string {
  if (tag.precision === 'auto') return 'fp16 on GPU · fp32 on CPU'
  return tag.precision ?? '—'
}

/** The layers a tag is made of, for the tag page "Details" list. */
export function layersFor(model: Model, tag: Tag): Layer[] {
  if (tag.kind === 'router') {
    return [
      {
        kind: 'router',
        preview: `routes to ${(tag.routesTo ?? []).map((t) => `${model.name}:${t}`).join(' | ')} by detected script and language`,
      },
      { kind: 'license', preview: 'Apache License, Version 2.0, January 2004' },
    ]
  }
  const precision = tag.precision === 'auto' ? 'fp16 | fp32' : tag.precision
  const decision = `{"engine": "onnx", "family": "${model.name}", "encoder": "${tag.encoder}", "max_len": ${tag.context}, …}`
  return [
    {
      kind: 'model',
      preview: `onnx · ${tag.backbone} · ${tag.params} · ${precision}`,
      size: sizeLabel(tag),
    },
    { kind: 'tokenizer', preview: 'tokenizer.json (Hugging Face tokenizers format)' },
    { kind: 'decision', preview: decision },
    { kind: 'calibration', preview: LAYA_CAL_PREVIEW },
    { kind: 'license', preview: 'Apache License, Version 2.0, January 2004' },
  ]
}

// ---------------------------------------------------------------------------------------------
// Search (runs in the browser: public/static/app.js filters with these precomputed strings)

/** Lower-case, NFKD-normalized — app.js normalizes queries the same way. */
export const normalize = (s: string): string => s.toLowerCase().normalize('NFKD')

/** Everything a query can match for a model. */
export function haystack(model: Model): string {
  return normalize(
    [
      model.name,
      model.title,
      model.description,
      model.publisher.name,
      ...model.capabilities,
      ...model.keywords,
      ...model.tags.map((t) => `${model.name}:${t.name} ${t.name} ${t.backbone ?? ''} ${t.languages}`),
    ].join(' '),
  )
}

export interface SearchIndex {
  models: {
    name: string
    href: string
    description: string
    search: string
    caps: Capability[]
    rank: number
    updated: string
    /** Featured tags a query can jump to directly (routers excluded). */
    tags: { name: string; href: string; summary: string; search: string }[]
  }[]
}

/** The static search index served at /search.json (navbar typeahead). */
export function searchIndex(): SearchIndex {
  return {
    models: [...catalog]
      .sort((a, b) => a.rank - b.rank)
      .map((m) => ({
        name: m.name,
        href: `/library/${m.name}`,
        description: m.description,
        search: haystack(m),
        caps: m.capabilities,
        rank: m.rank,
        updated: m.updated,
        tags: featuredTags(m)
          .filter((t) => t.kind !== 'router')
          .map((t) => ({
            name: fullName(m, t),
            href: `/library/${fullName(m, t)}`,
            summary: t.summary,
            search: normalize(`${fullName(m, t)} ${t.languages} ${t.backbone ?? ''}`),
          })),
      })),
  }
}
