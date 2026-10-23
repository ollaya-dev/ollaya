import type { Child } from 'hono/jsx'
import { Icon } from '../components/Icon'
import { CapBadge, CodeBlock, CodeTabs, OutlinePill, SizeBadge, textLink } from '../components/ui'
import { Updated } from '../components/Updated'
import {
  featuredTags,
  fullName,
  getTag,
  layersFor,
  precisionLabel,
  sizeLabel,
  type Model,
  type Tag,
} from '../data/catalog'
import { usageTabs, visionUsageTabs } from '../data/examples'
import { readmes } from '../generated/content'
import { benchGpu, measured, trainedOnTyped, withResults } from '../data/numbers'

// Pages: /library/<model>, /library/<model>/tags and /library/<model>:<tag>.
// (/library itself redirects to /search; see public/_redirects and LibraryRedirect below.)

// ---------------------------------------------------------------------------------------------

function Breadcrumb({ model, current }: { model: Model; current?: string }) {
  return (
    <nav aria-label="Breadcrumb" class="text-[13px] text-muted">
      <ol class="flex flex-wrap items-center gap-1.5">
        <li>
          <a href="/search" class="underline-offset-4 hover:text-fg hover:underline">
            Models
          </a>
        </li>
        <li aria-hidden="true">/</li>
        <li>
          {current ? (
            <a href={`/library/${model.name}`} class="underline-offset-4 hover:text-fg hover:underline">
              {model.name}
            </a>
          ) : (
            <span aria-current="page">{model.name}</span>
          )}
        </li>
        {current ? (
          <>
            <li aria-hidden="true">/</li>
            <li>
              <span aria-current="page">{current}</span>
            </li>
          </>
        ) : null}
      </ol>
    </nav>
  )
}

function MetaLine({ model, extra }: { model: Model; extra?: string[] }) {
  const items = [...(extra ?? [])]
  return (
    <p class="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-muted">
      <span class="inline-flex items-center gap-1.5">
        <Icon name="tag" class="size-4" />
        <a href={`/library/${model.name}/tags`} class="underline-offset-4 hover:text-fg hover:underline">
          {model.tags.length} Tags
        </a>
      </span>
      {model.updated ? <Updated iso={model.updated} /> : null}
      {items.map((i) => (
        <span>{i}</span>
      ))}
      {model.license ? <span>{model.license}</span> : null}
      {model.publisher ? (
        <span>
          by{' '}
          <a href={model.publisher.url} class="underline-offset-4 hover:text-fg hover:underline">
            {model.publisher.name}
          </a>
        </span>
      ) : null}
    </p>
  )
}

function ModelHeader({ model, tag, crumb }: { model: Model; tag?: Tag; crumb?: string }) {
  const caps = tag ? tag.capabilities : model.capabilities
  const sizes = tag ? (tag.params ? [tag.params.toLowerCase()] : []) : model.sizes
  return (
    <header>
      <Breadcrumb model={model} current={crumb} />
      <h1 class="mt-4 font-mono text-4xl leading-tight font-semibold tracking-tight break-words text-fg md:text-5xl">
        {tag ? fullName(model, tag) : model.name}
      </h1>
      <MetaLine
        model={model}
        extra={
          tag && tag.kind === 'model'
            ? [tag.params ? `${tag.params} params` : '', tag.context ? `${tag.context} context` : '', tag.languages].filter(Boolean)
            : undefined
        }
      />
      <p class="mt-4 text-base text-body md:text-lg">{tag ? tag.summary : model.description}</p>
      {caps.length || sizes.length ? (
        <div class="mt-4 flex flex-wrap gap-2">
          {caps.map((cap) => (
            <CapBadge>{cap}</CapBadge>
          ))}
          {sizes.map((s) => (
            <SizeBadge>{s}</SizeBadge>
          ))}
        </div>
      ) : null}
    </header>
  )
}

function Usage({ refName, model, tag }: { refName: string; model: Model; tag: Tag | undefined }) {
  const tabs = tag?.capabilities.includes('vision')
    ? visionUsageTabs(refName)
    : usageTabs(refName, model.exampleState ?? undefined, tag?.builtinQuestions ?? false, model.noPresets)
  return (
    <section class="mt-8" aria-label="Usage">
      <CodeTabs id="usage" label="Usage examples" tabs={tabs} />
    </section>
  )
}

const cell = 'px-4 py-3 align-top'
const head = 'px-4 py-2.5 font-medium'

function TagName({ model, tag }: { model: Model; tag: Tag }) {
  const isLatest = tag.name === 'latest'
  return (
    <span class="inline-flex flex-wrap items-center gap-2">
      <a
        href={`/library/${fullName(model, tag)}`}
        class="font-medium break-all text-fg underline-offset-4 hover:underline"
      >
        {isLatest ? model.name : fullName(model, tag)}
      </a>
      {isLatest ? <OutlinePill>latest</OutlinePill> : null}
    </span>
  )
}

const inputLabel = (tag: Tag) =>
  tag.kind === 'router'
    ? `Text · auto (${(tag.routesTo ?? []).join(' / ')})`
    : `${tag.capabilities.includes('vision') ? 'Text, image' : 'Text'} · ${tag.languages}`

/** A tag's bench numbers, for the tables: typed-decisions accuracy (not for a tag trained on that split) and the GPU latency. */
function tagNumbers(model: Model, tag: Tag) {
  // `latest` is usually an alias ("Same as decider:2b."); its numbers are the tag it names.
  const alias = tag.name === 'latest' ? tag.summary.match(/^Same as [^:]+:([^.\s]+)/)?.[1] : undefined
  const ref = fullName(model, (alias && getTag(model, alias)) || tag)
  return { acc: trainedOnTyped(ref) ? undefined : measured(ref, 'typed.accuracy'), ms: measured(ref, 'gpu.p50_ms') }
}

/**
 * The featured tags in groups a reader chooses by: the fastest, the most accurate, the ones that read
 * images, the router. A tag is in one group; within a group the order is the group's measure.
 */
function groupTags(model: Model): { title: string; note: string; tags: Tag[] }[] {
  const tags = featuredTags(model).map((t) => ({ t, ...tagNumbers(model, t) }))
  const routers = tags.filter((x) => x.t.kind === 'router')
  const vision = tags.filter((x) => x.t.kind !== 'router' && x.t.capabilities.includes('vision'))
  const text = tags.filter((x) => x.t.kind !== 'router' && !x.t.capabilities.includes('vision'))
  const measuredText = text.filter((x) => x.acc !== undefined && x.ms !== undefined)
  // With two or more measured text tags, the most accurate leads one group and the rest sort by speed.
  if (measuredText.length >= 2) {
    const best = [...measuredText].sort((p, q) => q.acc! - p.acc!)[0]!
    const rest = text.filter((x) => x !== best).sort((p, q) => (p.ms ?? Infinity) - (q.ms ?? Infinity) || (p.t.name === 'latest' ? 1 : 0) - (q.t.name === 'latest' ? 1 : 0))
    return [
      { title: 'Most accurate', note: 'Highest typed-decisions accuracy of the family.', tags: [best.t] },
      { title: 'Faster', note: 'Fastest first; accuracy as measured on the bench.', tags: rest.map((x) => x.t) },
      ...(vision.length ? [{ title: 'Reads images', note: 'A PNG with the state, through the author’s projector.', tags: vision.map((x) => x.t) }] : []),
      ...(routers.length ? [{ title: 'Router', note: 'Picks one of the tags above per request.', tags: routers.map((x) => x.t) }] : []),
    ].filter((g) => g.tags.length)
  }
  return [
    { title: 'Models', note: '', tags: text.sort((p, q) => (p.ms ?? Infinity) - (q.ms ?? Infinity)).map((x) => x.t) },
    ...(vision.length ? [{ title: 'Reads images', note: '', tags: vision.map((x) => x.t) }] : []),
    ...(routers.length ? [{ title: 'Router', note: '', tags: routers.map((x) => x.t) }] : []),
  ].filter((g) => g.tags.length)
}

const num = (v: number | undefined, digits: number, unit = '') => (v === undefined ? '—' : `${v.toFixed(digits)}${unit}`)

function ModelsTable({ model }: { model: Model }) {
  const groups = groupTags(model)
  const anyNumbers = featuredTags(model).some((t) => {
    const n = tagNumbers(model, t)
    return n.acc !== undefined || n.ms !== undefined
  })
  return (
    <section class="mt-12" aria-labelledby="models-title">
      <div class="flex items-baseline justify-between gap-4">
        <h2 id="models-title" class="text-xl font-semibold tracking-[-0.02em] text-fg">
          Models
        </h2>
        <a href={`/library/${model.name}/tags`} class="inline-flex items-center gap-1 text-[13px] text-muted underline-offset-4 hover:text-fg hover:underline">
          Every tag <Icon name="arrowRight" class="size-3.5" />
        </a>
      </div>
      <div class="mt-4 overflow-hidden rounded-md border border-line">
        <table class="w-full text-left text-sm">
          <thead class="bg-subtle text-[13px] text-muted">
            <tr>
              <th scope="col" class={head}>
                Name
              </th>
              {anyNumbers ? (
                <>
                  <th scope="col" class={`${head} text-right`}>
                    Accuracy
                  </th>
                  <th scope="col" class={`${head} text-right`}>
                    Latency
                  </th>
                </>
              ) : null}
              <th scope="col" class={`${head} hidden text-right sm:table-cell`}>
                Size
              </th>
              <th scope="col" class={`${head} hidden text-right sm:table-cell`}>
                Context
              </th>
              <th scope="col" class={`${head} hidden lg:table-cell`}>
                Input
              </th>
            </tr>
          </thead>
          {groups.map((g) => (
            <tbody class="border-t border-line">
              {groups.length > 1 ? (
                <tr class="bg-subtle/60">
                  <th scope="rowgroup" colspan={6} class="px-4 py-2 text-left text-[13px] font-medium text-fg">
                    {g.title}
                    {g.note ? <span class="ml-2 font-normal text-muted">{g.note}</span> : null}
                  </th>
                </tr>
              ) : null}
              {g.tags.map((t) => {
                const n = tagNumbers(model, t)
                return (
                  <tr class="border-t border-line">
                    <td class={cell}>
                      <TagName model={model} tag={t} />
                      <span class="mt-1 block text-[13px] text-muted">{t.summary}</span>
                    </td>
                    {anyNumbers ? (
                      <>
                        <td class={`${cell} text-right whitespace-nowrap text-fg tabular-nums`}>{n.acc === undefined && trainedOnTyped(fullName(model, t)) ? <span class="text-muted" title="Trained on this dataset's train split">trained</span> : num(n.acc, 3)}</td>
                        <td class={`${cell} text-right whitespace-nowrap text-fg tabular-nums`}>{n.ms === undefined ? '—' : `${Math.round(n.ms)} ms`}</td>
                      </>
                    ) : null}
                    <td class={`${cell} hidden text-right whitespace-nowrap text-body tabular-nums sm:table-cell`}>{sizeLabel(t)}</td>
                    <td class={`${cell} hidden text-right whitespace-nowrap text-body tabular-nums sm:table-cell`}>{t.context}</td>
                    <td class={`${cell} hidden text-body lg:table-cell`}>{inputLabel(t)}</td>
                  </tr>
                )
              })}
            </tbody>
          ))}
        </table>
      </div>
      <p class="mt-2 text-[13px] text-muted">
        {anyNumbers ? `Accuracy: typed-decisions, argmax against the majority label on 400 states (left out for a tag trained on that data). Latency: the median five-question request on an ${benchGpu}, Ollaya Bench. ` : ''}
        {featuredTags(model).some((t) => t.precision === 'auto') ? 'Each model carries fp16 and fp32 graphs over one weights file, and loads fp16 on a CUDA GPU and fp32 on CPU.' : ''}
      </p>
    </section>
  )
}

/** The right column of a model's pages: the numbers and facts a reader scans before reading. */
function Aside({ model, tag }: { model: Model; tag?: Tag }) {
  const main = tag ?? getTag(model, 'latest') ?? model.tags[0]
  const n = main ? tagNumbers(model, main) : { acc: undefined, ms: undefined }
  const rows: [string, Child][] = [
    ['Publisher', model.publisher ? <a href={model.publisher.url} class={textLink}>{model.publisher.name}</a> : '—'],
    ['License', model.license || '—'],
    ['Sizes', model.sizes.join(' · ') || '—'],
    ['Tags', <a href={`/library/${model.name}/tags`} class={textLink}>{model.tags.length}</a>],
    ...(model.updated ? ([['Updated', <Updated iso={model.updated} />]] as [string, Child][]) : []),
    ...(model.source ? ([['Weights', <a href={model.source} class={textLink}>source repository</a>]] as [string, Child][]) : []),
  ]
  return (
    <aside class="lg:sticky lg:top-24" aria-label="At a glance">
      <div class="rounded-md border border-line">
        {n.acc !== undefined || n.ms !== undefined ? (
          <dl class="grid grid-cols-2 divide-x divide-line border-b border-line">
            <div class="p-5">
              <dd class="text-3xl font-semibold tracking-[-0.02em] text-fg tabular-nums">{num(n.acc, 3)}</dd>
              <dt class="mt-1 text-[13px] text-muted">accuracy, typed decisions</dt>
            </div>
            <div class="p-5">
              <dd class="text-3xl font-semibold tracking-[-0.02em] text-fg tabular-nums">{n.ms === undefined ? '—' : `${Math.round(n.ms)} ms`}</dd>
              <dt class="mt-1 text-[13px] text-muted">five questions, {benchGpu}</dt>
            </div>
          </dl>
        ) : null}
        <dl class="divide-y divide-line text-sm">
          {rows.map(([k, v]) => (
            <div class="flex items-baseline justify-between gap-4 px-5 py-3">
              <dt class="text-muted">{k}</dt>
              <dd class="text-right text-body">{v}</dd>
            </div>
          ))}
        </dl>
        {main ? (
          <div class="border-t border-line p-5">
            <p class="text-[13px] text-muted">Run it</p>
            <CodeBlock code={`ollaya run ${main.name === 'latest' ? model.name : fullName(model, main)}`} class="mt-2" />
          </div>
        ) : null}
      </div>
    </aside>
  )
}

/** The page shell shared by a model's pages: the content on the left, the aside on the right. */
function Shell({ aside, children }: { aside: Child; children: Child }) {
  return (
    <div class="px-6 pt-10 pb-16 md:px-10 md:pt-14 lg:grid lg:grid-cols-[minmax(0,1fr)_20rem] lg:gap-14">
      <div class="min-w-0">{children}</div>
      <div class="mt-12 lg:mt-0">{aside}</div>
    </div>
  )
}

export function ModelPage({ model, origin }: { model: Model; origin: string }) {
  const readme = readmes[model.name]
  const badge = `[![Run on Ollaya](${origin}/badge/run-on-ollaya.svg)](${origin}/library/${model.name})`
  return (
    <Shell aside={<Aside model={model} />}>
      <ModelHeader model={model} />
      <Usage refName={model.name} model={model} tag={getTag(model, 'latest') ?? model.tags[0]} />
      <ModelsTable model={model} />
      {readme ? (
        <section class="mt-14 border-t border-line pt-10" aria-labelledby="readme-title">
          <h2 id="readme-title" class="text-xl font-semibold tracking-[-0.02em] text-fg">
            Readme
          </h2>
          <div class="prose mt-6" dangerouslySetInnerHTML={{ __html: withResults(readme) }}></div>
        </section>
      ) : null}
      <section class="mt-14 border-t border-line pt-10" aria-labelledby="badge-title">
        <h2 id="badge-title" class="text-xl font-semibold tracking-[-0.02em] text-fg">
          Badge
        </h2>
        <img src="/badge/run-on-ollaya.svg" alt="Run on Ollaya" width="104" height="20" class="mt-4" />
        <p class="mt-3 text-sm text-body">
          For the model card or a README. <a href="/docs/model-authors" class={textLink}>Day-0 support for a new model</a>
        </p>
        <CodeBlock code={badge} lang="markdown" class="mt-3" />
      </section>
    </Shell>
  )
}

export function TagsPage({ model }: { model: Model }) {
  return (
    <Shell aside={<Aside model={model} />}>
      <ModelHeader model={model} crumb="tags" />
      <section class="mt-10" aria-labelledby="tags-title">
        <h2 id="tags-title" class="text-base font-semibold text-fg">
          Tags
        </h2>
        <div class="mt-3 overflow-hidden rounded-md border border-line">
          <table class="w-full text-left text-sm">
            <thead class="bg-subtle text-[13px] text-muted">
              <tr>
                <th scope="col" class={head}>
                  Name
                </th>
                <th scope="col" class={`${head} hidden md:table-cell`}>
                  Precision
                </th>
                <th scope="col" class={`${head} hidden sm:table-cell`}>
                  Size
                </th>
                <th scope="col" class={`${head} hidden sm:table-cell`}>
                  Context
                </th>
                <th scope="col" class={`${head} hidden lg:table-cell`}>
                  Input
                </th>
              </tr>
            </thead>
            <tbody class="divide-y divide-line">
              {model.tags.map((t) => (
                <tr>
                  <td class={cell}>
                    <TagName model={model} tag={t} />
                    <span class="mt-1 block text-[13px] text-muted">
                      {t.variantOf ? `${t.precision} variant of ${model.name}:${t.variantOf}` : t.summary}
                    </span>
                    <span class="mt-1 block text-[13px] text-muted sm:hidden">
                      {t.kind === 'router' ? 'router' : `${sizeLabel(t)} · ${t.context} ctx · ${t.languages}`}
                    </span>
                  </td>
                  <td class={`${cell} hidden text-body md:table-cell`}>{precisionLabel(t)}</td>
                  <td class={`${cell} hidden whitespace-nowrap text-body tabular-nums sm:table-cell`}>
                    {sizeLabel(t)}
                  </td>
                  <td class={`${cell} hidden whitespace-nowrap text-body tabular-nums sm:table-cell`}>{t.context}</td>
                  <td class={`${cell} hidden text-body lg:table-cell`}>{inputLabel(t)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p class="mt-2 text-[13px] text-muted">
          Tags without a suffix load fp16 on a CUDA GPU and fp32 on CPU; add <code class="font-mono">-fp16</code> or{' '}
          <code class="font-mono">-fp32</code> to pin one precision.
        </p>
      </section>
    </Shell>
  )
}

export function TagPage({ model, tag }: { model: Model; tag: Tag }) {
  const ref = fullName(model, tag)
  const layers = layersFor(model, tag)
  const variants = model.tags.filter((t) => t.variantOf === tag.name)
  const base = tag.variantOf ? getTag(model, tag.variantOf) : undefined
  return (
    <Shell aside={<Aside model={model} tag={tag} />}>
      <ModelHeader model={model} tag={tag} crumb={tag.name} />
      <Usage refName={tag.name === 'latest' ? model.name : ref} model={model} tag={tag} />

      <section class="mt-10" aria-labelledby="details-title">
        <h2 id="details-title" class="text-base font-semibold text-fg">
          Details
        </h2>
        <ul class="mt-3 divide-y divide-line rounded-md border border-line" role="list">
          {layers.map((l) => (
            <li class="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-4 gap-y-1 px-4 py-3 sm:grid-cols-[7rem_minmax(0,1fr)_auto]">
              <span class="text-sm font-medium text-fg">{l.kind}</span>
              <span class="text-right text-[13px] whitespace-nowrap text-muted tabular-nums sm:order-last">
                {l.digest ? (
                  <span class="font-mono text-[11px]" title={`sha256:${l.digest}…`}>
                    {l.digest}
                  </span>
                ) : null}
                {l.digest ? ' · ' : ''}
                {l.size}
              </span>
              <code
                class="col-span-2 font-mono text-[13px] break-words text-muted sm:col-span-1 sm:truncate"
                title={l.preview}
              >
                {l.preview}
              </code>
            </li>
          ))}
        </ul>
        <p class="mt-2 text-[13px] text-muted">
          Every layer is checked against its sha256 when it is pulled. Weights and tokenizers download from the
          model author's Hugging Face repository at a pinned commit; Ollaya never re-hosts them.
        </p>
      </section>

      {tag.routesTo?.length ? (
        <section class="mt-10" aria-labelledby="routes-title">
          <h2 id="routes-title" class="text-base font-semibold text-fg">
            Routes to
          </h2>
          <ul class="mt-3 space-y-2 text-sm" role="list">
            {tag.routesTo.map((name) => {
              const t = getTag(model, name)
              return t ? (
                <li>
                  <a href={`/library/${fullName(model, t)}`} class={`font-medium ${textLink}`}>
                    {fullName(model, t)}
                  </a>{' '}
                  <span class="text-muted">· {t.summary}</span>
                </li>
              ) : null
            })}
          </ul>
        </section>
      ) : null}

      {variants.length || base ? (
        <section class="mt-10" aria-labelledby="variants-title">
          <h2 id="variants-title" class="text-base font-semibold text-fg">
            {base ? 'Variant of' : 'Precision variants'}
          </h2>
          <ul class="mt-3 flex flex-wrap gap-2" role="list">
            {(base ? [base] : variants).map((t) => (
              <li>
                <a
                  href={`/library/${fullName(model, t)}`}
                  class="inline-flex items-center gap-2 rounded-md border border-line px-3 py-1 text-[13px] text-body hover:border-line-strong hover:text-fg"
                >
                  <span class="font-medium">{fullName(model, t)}</span>
                  <span class="text-muted">{sizeLabel(t)}</span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <p class="mt-10 text-sm">
        <a
          href={`/library/${model.name}`}
          class="inline-flex items-center gap-1.5 font-medium text-fg underline-offset-4 hover:underline"
        >
          {model.title} readme and all models <Icon name="arrowRight" class="size-4" />
        </a>
      </p>
    </Shell>
  )
}

/**
 * /library → /search. Cloudflare applies the 302 from public/_redirects before this file is
 * reached; other static hosts serve this page, which forwards with a meta refresh.
 */
export function LibraryRedirect({ origin }: { origin: string }) {
  return (
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <title>Models · Ollaya</title>
        <meta name="robots" content="noindex" />
        <link rel="canonical" href={`${origin}/search`} />
        <meta http-equiv="refresh" content="0; url=/search" />
      </head>
      <body>
        <p>
          Moved to <a href="/search">/search</a>.
        </p>
      </body>
    </html>
  )
}
