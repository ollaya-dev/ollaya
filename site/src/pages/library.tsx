import { Icon } from '../components/Icon'
import { CapBadge, CodeTabs, OutlinePill, SizeBadge, textLink } from '../components/ui'
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
import { usageTabs } from '../data/examples'
import { readmes } from '../generated/content'

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
      <h1 class="mt-4 text-[28px] leading-tight font-medium tracking-tight break-words text-fg">
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
  const tabs = usageTabs(refName, model.exampleState ?? undefined, tag?.builtinQuestions ?? false)
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
  tag.kind === 'router' ? `Text · auto (${(tag.routesTo ?? []).join(' / ')})` : `Text · ${tag.languages}`

function ModelsTable({ model }: { model: Model }) {
  return (
    <section class="mt-10" aria-labelledby="models-title">
      <div class="flex items-baseline justify-between gap-4">
        <h2 id="models-title" class="text-base font-semibold text-fg">
          Models
        </h2>
        <a
          href={`/library/${model.name}/tags`}
          class="inline-flex items-center gap-1 text-[13px] text-muted underline-offset-4 hover:text-fg hover:underline"
        >
          View all <Icon name="arrowRight" class="size-3.5" />
        </a>
      </div>
      <div class="mt-3 overflow-hidden rounded-lg border border-line">
        <table class="w-full text-left text-sm">
          <thead class="bg-subtle text-[13px] text-muted">
            <tr>
              <th scope="col" class={head}>
                Name
              </th>
              <th scope="col" class={`${head} hidden sm:table-cell`}>
                Size
              </th>
              <th scope="col" class={`${head} hidden sm:table-cell`}>
                Context
              </th>
              <th scope="col" class={`${head} hidden md:table-cell`}>
                Input
              </th>
            </tr>
          </thead>
          <tbody class="divide-y divide-line">
            {featuredTags(model).map((t) => (
              <tr>
                <td class={cell}>
                  <TagName model={model} tag={t} />
                  <span class="mt-1 block text-[13px] text-muted sm:hidden">
                    {t.kind === 'router' ? 'router' : `${sizeLabel(t)} · ${t.context} ctx · ${t.languages}`}
                  </span>
                </td>
                <td class={`${cell} hidden whitespace-nowrap text-body tabular-nums sm:table-cell`}>{sizeLabel(t)}</td>
                <td class={`${cell} hidden whitespace-nowrap text-body tabular-nums sm:table-cell`}>{t.context}</td>
                <td class={`${cell} hidden text-body md:table-cell`}>{inputLabel(t)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p class="mt-2 text-[13px] text-muted">
        Each model carries fp16 and fp32 graphs over one weights file, and loads fp16 on a CUDA GPU and fp32 on CPU.
      </p>
    </section>
  )
}

export function ModelPage({ model }: { model: Model }) {
  const readme = readmes[model.name]
  return (
    <div class="mx-auto max-w-[52rem] px-4 pt-8 md:px-6 md:pt-12">
      <ModelHeader model={model} />
      <Usage refName={model.name} model={model} tag={getTag(model, 'latest') ?? model.tags[0]} />
      <ModelsTable model={model} />
      {readme ? (
        <section class="mt-12 border-t border-line pt-10" aria-labelledby="readme-title">
          <h2 id="readme-title" class="text-base font-semibold text-fg">
            Readme
          </h2>
          <div class="prose mt-6" dangerouslySetInnerHTML={{ __html: readme }}></div>
        </section>
      ) : null}
    </div>
  )
}

export function TagsPage({ model }: { model: Model }) {
  return (
    <div class="mx-auto max-w-[52rem] px-4 pt-8 md:px-6 md:pt-12">
      <ModelHeader model={model} crumb="tags" />
      <section class="mt-10" aria-labelledby="tags-title">
        <h2 id="tags-title" class="text-base font-semibold text-fg">
          Tags
        </h2>
        <div class="mt-3 overflow-hidden rounded-lg border border-line">
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
    </div>
  )
}

export function TagPage({ model, tag }: { model: Model; tag: Tag }) {
  const ref = fullName(model, tag)
  const layers = layersFor(model, tag)
  const variants = model.tags.filter((t) => t.variantOf === tag.name)
  const base = tag.variantOf ? getTag(model, tag.variantOf) : undefined
  return (
    <div class="mx-auto max-w-[52rem] px-4 pt-8 md:px-6 md:pt-12">
      <ModelHeader model={model} tag={tag} crumb={tag.name} />
      <Usage refName={tag.name === 'latest' ? model.name : ref} model={model} tag={tag} />

      <section class="mt-10" aria-labelledby="details-title">
        <h2 id="details-title" class="text-base font-semibold text-fg">
          Details
        </h2>
        <ul class="mt-3 divide-y divide-line rounded-lg border border-line" role="list">
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
                  class="inline-flex items-center gap-2 rounded-3xl border border-line px-3 py-1 text-[13px] text-body hover:border-line-strong hover:text-fg"
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
    </div>
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
