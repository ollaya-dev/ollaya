/**
 * The model catalog: facts from the static registry, plus a few hand-written extras.
 *
 * Single source of truth: ../registry (manifests and config blobs written by
 * convert/ollaya_convert/package.py), read at build time by scripts/gen-registry.mjs into
 * src/generated/registry.ts. Adding a model to the registry adds it to the site; an overlay
 * below and a readme in content/library/<model>.md are optional.
 *
 * Rules: never add pull counts or other numbers we do not have.
 */
import { registryModels, type RegistryModel, type RegistryTag } from '../generated/registry'

// ---------------------------------------------------------------------------------------------
// Hand-written extras (optional per model)

interface TagOverlay {
  summary?: string
  capabilities?: string[]
}

interface ModelOverlay {
  title?: string
  description?: string
  publisher?: { name: string; url: string }
  capabilities?: string[]
  keywords?: string[]
  /** Curated order for "Popular" (we have no pull counts, so this is editorial). Lower first. */
  rank?: number
  tags?: Record<string, TagOverlay>
}

const overlays: Record<string, ModelOverlay> = {
  laya: {
    title: 'Laya',
    description:
      'Open decision models from Convai Innovations. Typed, calibrated answers to choice, score and yes/no questions in a single forward pass, in English and 100+ languages.',
    publisher: { name: 'Convai Innovations', url: 'https://huggingface.co/convaiinnovations' },
    capabilities: ['multilingual', 'router', 'guardrails', 'fine-tuned'],
    keywords: ['decision', 'classification', 'classifier', 'triage', 'moderation', 'guardrail', 'routing', 'typesafe', 'jev', 'system one'],
    rank: 1,
    tags: {
      latest: { summary: 'Router: sends English text to laya:en and everything else to laya:multilingual.' },
      en: { summary: 'English. Best for guardrails and email triage.', capabilities: ['guardrails'] },
      multilingual: {
        summary: '100+ languages, 1024-token context; up to ~2.2× faster on batched calls.',
        capabilities: ['multilingual'],
      },
      'typed-decisions': {
        summary: 'Fine-tuned on typed-decisions workflows: 0.766 accuracy vs 0.727 published for Jev 1.13.',
        capabilities: ['fine-tuned'],
      },
    },
  },
}

/**
 * Upcoming families, shown as one "planned" line. An entry disappears automatically once a model
 * whose name contains `key` is in the registry.
 */
const planned: { key: string; label: string }[] = [
  { key: 'von', label: 'von' },
  { key: 'gliclass', label: 'GLiClass' },
  { key: 'nli', label: 'NLI zero-shot classifiers' },
  { key: 'decider', label: 'decider' },
  { key: 'gguf', label: 'GGUF LLM-based decision models via llama.cpp' },
]

// ---------------------------------------------------------------------------------------------
// Types used by the pages

export type Capability = string
export type Precision = 'auto' | 'fp16' | 'fp32'

/** A layer as shown in a tag page's "Details" list. */
export interface Layer {
  kind: string
  preview: string
  size: string
  digest: string | null
}

export interface Tag {
  /** Tag name without the model, e.g. "en" or "en-fp32". */
  name: string
  kind: 'router' | 'model'
  /** Base tag for precision-pinned variants ("en" for "en-fp16"). */
  variantOf?: string
  precision: Precision | null
  /** Tags the router may send a request to (router tags only). */
  routesTo?: string[]
  backbone?: string
  encoder?: string
  params?: string
  /** Context window in tokens, as text ("512"); routers show their targets' range. */
  context: string
  languages: string
  sizeBytes: number
  summary: string
  /** Shown in the model page "Models" table: every tag that is not a precision variant. */
  featured: boolean
  capabilities: Capability[]
  releaseDate: string | null
  /** Short model digest, as `ollaya list` prints it. */
  id: string
  registry: RegistryTag
}

export interface Model {
  name: string
  title: string
  description: string
  publisher: { name: string; url: string } | null
  /** Link to the weights' source repository, if known. */
  source: string | null
  license: string
  capabilities: Capability[]
  /** Parameter-size badges, lower case like "421m". */
  sizes: string[]
  /** Latest release date of any tag (YYYY-MM-DD), if known. */
  updated: string | null
  rank: number
  keywords: string[]
  tags: Tag[]
}

// ---------------------------------------------------------------------------------------------
// Building the catalog

const LANGUAGE_NAMES: Record<string, string> = {
  en: 'English',
  multilingual: '100+ languages',
  tr: 'Turkish',
  de: 'German',
  fr: 'French',
  es: 'Spanish',
}

function languagesLabel(langs: string[]): string {
  if (!langs.length) return '—'
  return langs.map((l) => LANGUAGE_NAMES[l] ?? l).join(', ')
}

/** "huggingface.co/org/repo@commit" → https://huggingface.co/org/repo */
function sourceUrl(source: string): string | null {
  const m = /^(?:https?:\/\/)?(huggingface\.co\/[^@\s]+)/.exec(source)
  return m ? `https://${m[1]}` : null
}

function publisherFrom(source: string): { name: string; url: string } | null {
  const m = /^(?:https?:\/\/)?huggingface\.co\/([^/@\s]+)/.exec(source)
  return m ? { name: m[1]!, url: `https://huggingface.co/${m[1]}` } : null
}

function buildTag(model: RegistryModel, t: RegistryTag, overlay: ModelOverlay | undefined): Omit<Tag, 'featured' | 'variantOf'> {
  const isRouter = t.config.format === 'router' || t.router !== null
  const o = overlay?.tags?.[t.name] ?? (t.precision ? overlay?.tags?.[t.name.replace(/-(fp16|fp32)$/, '')] : undefined)
  const graphs = new Set(t.layers.filter((l) => l.kind === 'graph').map((l) => l.precision))
  const precision: Precision | null = isRouter
    ? null
    : t.precision === 'fp16' || t.precision === 'fp32'
      ? t.precision
      : graphs.size === 1
        ? ((graphs.values().next().value as Precision) ?? 'auto')
        : 'auto'
  const routesTo = t.router
    ? Object.values(t.router.routes).map((target) => target.replace(new RegExp(`^${model.name}:`), ''))
    : undefined
  const caps = new Set<string>(o?.capabilities ?? [])
  if (isRouter) caps.add('router')
  if (t.config.languages.includes('multilingual')) caps.add('multilingual')
  return {
    name: t.name,
    kind: isRouter ? 'router' : 'model',
    precision,
    routesTo,
    backbone: t.encoder ? t.encoder.split('/').pop() : undefined,
    encoder: t.encoder ?? undefined,
    params: t.config.parameterSize || undefined,
    context: t.config.contextLength ? String(t.config.contextLength) : '',
    languages: isRouter ? 'Auto-detected' : languagesLabel(t.config.languages),
    sizeBytes: t.size,
    summary: o?.summary ?? t.config.description,
    capabilities: [...caps],
    releaseDate: t.config.releaseDate,
    id: t.digest.slice(0, 12),
    registry: t,
  }
}

function buildModel(m: RegistryModel): Model {
  const overlay = overlays[m.name]
  const base = m.tags.map((t) => buildTag(m, t, overlay))
  const names = new Set(base.map((t) => t.name))
  const tags: Tag[] = base.map((t) => {
    const stem = t.name.replace(/-(fp16|fp32)$/, '')
    const variantOf = t.registry.precision && stem !== t.name && names.has(stem) ? stem : undefined
    return { ...t, variantOf, featured: !variantOf }
  })
  // Routers resolve their context from their targets.
  for (const t of tags) {
    if (t.kind !== 'router' || t.context) continue
    const ctx = [...new Set((t.routesTo ?? []).map((n) => tags.find((x) => x.name === n)?.context).filter(Boolean))]
    t.context = ctx.join(' / ')
  }
  // latest first, then bare tags, then variants, each alphabetically.
  const order = (t: Tag) => (t.name === 'latest' ? 0 : t.variantOf ? 2 : 1)
  tags.sort((a, b) => order(a) - order(b) || (a.variantOf ?? a.name).localeCompare(b.variantOf ?? b.name) || a.name.localeCompare(b.name))

  const models = tags.filter((t) => t.kind === 'model')
  const first = tags.find((t) => t.name === 'latest') ?? tags[0]!
  const source = models.map((t) => t.registry.config.source).find(Boolean) ?? ''
  const caps = new Set<string>(overlay?.capabilities ?? [])
  for (const t of tags) for (const c of t.capabilities) caps.add(c)
  const dates = tags.map((t) => t.releaseDate).filter((d): d is string => !!d).sort()
  return {
    name: m.name,
    title: overlay?.title ?? m.name,
    description: overlay?.description ?? first.registry.config.description,
    publisher: overlay?.publisher ?? publisherFrom(source),
    source: sourceUrl(source),
    license: first.registry.config.license || models[0]?.registry.config.license || '',
    capabilities: [...caps],
    sizes: [...new Set(models.map((t) => t.params?.toLowerCase()).filter((s): s is string => !!s))].sort(
      (a, b) => parseFloat(a) - parseFloat(b),
    ),
    updated: dates.at(-1) ?? null,
    rank: overlay?.rank ?? 100,
    keywords: overlay?.keywords ?? [],
    tags,
  }
}

/** Every model in the registry's `library` namespace, in curated order. */
export const catalog: Model[] = registryModels
  .filter((m) => m.namespace === 'library')
  .map(buildModel)
  .sort((a, b) => a.rank - b.rank || a.name.localeCompare(b.name))

/** Filter chips on /search: the capabilities present in the catalog, in a fixed order. */
const CAPABILITY_ORDER = ['multilingual', 'router', 'guardrails', 'fine-tuned']
export const capabilityFilters: { id: Capability; label: string }[] = [
  ...new Set(catalog.flatMap((m) => m.capabilities)),
]
  .sort((a, b) => (CAPABILITY_ORDER.indexOf(a) + 1 || 99) - (CAPABILITY_ORDER.indexOf(b) + 1 || 99) || a.localeCompare(b))
  .map((id) => ({ id, label: id }))

/** Planned families that are not in the registry yet. */
export const comingNext: string[] = planned
  .filter((p) => !catalog.some((m) => m.name.toLowerCase().includes(p.key)))
  .map((p) => p.label)

// ---------------------------------------------------------------------------------------------
// Lookups & formatting

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

/** Bytes as `ollaya list` prints them: "854 MB", "1.2 GB", "11 KB". */
export function formatBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`
  if (n >= 1e3) return `${Math.round(n / 1e3)} KB`
  return `${n} B`
}

/** Download size; routers show "—" because pulling one pulls its targets. */
export function sizeLabel(tag: Tag): string {
  return tag.kind === 'router' ? '—' : formatBytes(tag.sizeBytes)
}

export function precisionLabel(tag: Tag): string {
  if (tag.precision === 'auto') return 'fp16 on GPU · fp32 on CPU'
  return tag.precision ?? '—'
}

function shortDigest(d: string): string {
  return d.replace(/^sha256:/, '').slice(0, 12)
}

/** The layers of a tag, for its "Details" list. */
export function layersFor(model: Model, tag: Tag): Layer[] {
  const r = tag.registry
  const out: Layer[] = []
  for (const l of r.layers) {
    let preview: string
    switch (l.kind) {
      case 'graph':
        preview = `onnx · ${tag.backbone ?? r.config.family} · ${tag.params ?? ''} · ${l.precision ?? ''}`.replace(/ · (?= ·|$)/g, '')
        break
      case 'weights':
      case 'tokenizer':
        preview = l.url ? l.url.replace(/^https?:\/\//, '').replace(/\/resolve\/([0-9a-f]{7})[0-9a-f]*\//, '/resolve/$1…/') : ''
        break
      case 'decision':
        preview = `{"engine": "onnx", "family": "${r.config.family}", "encoder": "${r.encoder ?? ''}", "layout": "${r.layout ?? ''}", …}`
        break
      case 'calibration':
        preview = r.calibrationKeys?.length
          ? `{"temperature_by_options": {${r.calibrationKeys.map((k) => `"${k}": …`).join(', ')}}}`
          : '{"temperature": [1.0, 1.0, 1.0]}'
        break
      case 'router':
        preview = r.router
          ? `${r.router.strategy}: ${Object.entries(r.router.routes).map(([k, v]) => `${k} → ${v}`).join(', ')} (default ${r.router.default})`
          : 'router'
        break
      case 'params':
        preview = tag.precision ? `precision ${tag.precision}` : 'params'
        break
      case 'license':
        preview = r.licenseLine ?? r.config.license
        break
      default:
        preview = l.mediaType
    }
    out.push({ kind: l.kind, preview, size: formatBytes(l.size), digest: shortDigest(l.digest) })
  }
  return out
}

// ---------------------------------------------------------------------------------------------
// Search (runs in the browser: public/static/app.js filters with these precomputed strings)

export type SortKey = 'popular' | 'newest'

/** Lower-case, NFKD-normalized — app.js normalizes queries the same way. */
export const normalize = (s: string): string => s.toLowerCase().normalize('NFKD')

/** Everything a query can match for a model. */
export function haystack(model: Model): string {
  return normalize(
    [
      model.name,
      model.title,
      model.description,
      model.publisher?.name ?? '',
      ...model.capabilities,
      ...model.keywords,
      ...model.tags.map((t) => `${model.name}:${t.name} ${t.name} ${t.backbone ?? ''} ${t.languages} ${t.summary}`),
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
    updated: string | null
    /** Featured tags a query can jump to directly (routers excluded). */
    tags: { name: string; href: string; summary: string; search: string }[]
  }[]
}

/** The static search index served at /search.json (navbar typeahead). */
export function searchIndex(): SearchIndex {
  return {
    models: catalog.map((m) => ({
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
