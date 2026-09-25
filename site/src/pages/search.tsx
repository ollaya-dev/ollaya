import { Icon } from '../components/Icon'
import { CapBadge, SizeBadge } from '../components/ui'
import { Updated } from '../components/Updated'
import { capabilityFilters, catalog, comingNext, haystack, type Model } from '../data/catalog'

/**
 * /search: every model is rendered into the static page (so it works without JavaScript).
 * public/static/app.js filters and sorts the rows in place using their data-* attributes and
 * keeps ?q=&c=&o= in the URL with history.replaceState.
 */
export function SearchPage() {
  const models = [...catalog].sort((a, b) => a.rank - b.rank)
  return (
    <div class="mx-auto max-w-2xl px-4 pt-8 pb-8 md:px-6 md:pt-12">
      <h1 class="text-[28px] font-medium tracking-tight text-fg">Models</h1>
      <p class="mt-1 text-[15px] text-muted">Open decision models you can run locally with Ollaya.</p>
      <form id="search-form" action="/search" method="get" role="search" class="mt-6" data-search-form>
        <div class="relative">
          <label for="search-q" class="sr-only">
            Search models
          </label>
          <span class="pointer-events-none absolute inset-y-0 left-4 flex items-center text-muted">
            <Icon name="search" class="size-5" />
          </span>
          <input
            id="search-q"
            name="q"
            type="search"
            placeholder="Search models"
            autocomplete="off"
            spellcheck={false}
            class="w-full rounded-full border border-line bg-canvas py-2.5 pr-4 pl-12 text-base text-fg hover:border-line-strong focus:border-line-strong focus:ring-4 focus:ring-fill-strong focus:outline-none"
          />
        </div>
        <div class="mt-4 flex flex-wrap items-center justify-between gap-3">
          <fieldset class="flex flex-wrap gap-2">
            <legend class="sr-only">Filter by capability</legend>
            {capabilityFilters.map((f) => (
              <label class="relative inline-flex cursor-pointer items-center rounded-3xl border border-line px-3 py-1 text-[13px] text-body select-none hover:border-line-strong has-checked:border-fg has-checked:bg-fg has-checked:text-canvas has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-fg">
                <input type="checkbox" name="c" value={f.id} class="sr-only" />
                {f.label}
              </label>
            ))}
          </fieldset>
          <div class="flex items-center gap-2">
            <label for="search-sort" class="text-[13px] text-muted">
              Sort
            </label>
            <div class="relative">
              <select
                id="search-sort"
                name="o"
                class="appearance-none rounded-full border border-line bg-canvas py-1 pr-8 pl-3 text-[13px] text-fg hover:border-line-strong focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fg"
              >
                <option value="popular" selected>
                  Popular
                </option>
                <option value="newest">Newest</option>
              </select>
              <span class="pointer-events-none absolute inset-y-0 right-2.5 flex items-center text-muted">
                <Icon name="chevronDown" class="size-3.5" />
              </span>
            </div>
          </div>
        </div>
        <button
          type="submit"
          class="sr-only focus:not-sr-only focus:mt-4 focus:inline-flex focus:rounded-full focus:border focus:border-line focus:px-4 focus:py-1.5 focus:text-sm"
        >
          Search
        </button>
      </form>

      <div id="results" class="mt-6" aria-live="polite">
        <p class="sr-only" data-results-count>
          {models.length} {models.length === 1 ? 'model' : 'models'}
        </p>
        <ul class="divide-y divide-line border-t border-line" role="list" data-results-list>
          {models.map((m) => (
            <ModelRow model={m} />
          ))}
        </ul>
        <div class="border-t border-line py-12 text-center" data-results-empty hidden>
          <p class="text-fg">
            No models found<span data-results-query></span>.
          </p>
          <p class="mt-2 text-sm text-muted">
            <a href="/search" class="underline underline-offset-4 hover:text-fg" data-results-clear>
              Clear search and filters
            </a>
          </p>
        </div>
      </div>

      {comingNext.length ? (
        <aside class="mt-10 rounded-lg border border-line p-5" aria-labelledby="planned">
          <h2 id="planned" class="text-sm font-semibold text-fg">
            Planned
          </h2>
          <p class="mt-1.5 text-[15px] text-body">
            More open decision models are on the way: {comingNext.join(', ')}.
          </p>
        </aside>
      ) : null}
    </div>
  )
}

export function ModelRow({ model }: { model: Model }) {
  return (
    <li
      data-search={haystack(model)}
      data-caps={model.capabilities.join(' ')}
      data-rank={String(model.rank)}
      data-updated={model.updated ?? ''}
    >
      <a href={`/library/${model.name}`} class="group block py-6">
        <h2 class="text-xl font-medium text-fg underline-offset-4 group-hover:underline md:text-2xl">{model.name}</h2>
        <p class="mt-1.5 text-[15px] text-body md:text-base">{model.description}</p>
        <div class="mt-3 flex flex-wrap gap-2">
          {model.capabilities.map((cap) => (
            <CapBadge>{cap}</CapBadge>
          ))}
          {model.sizes.map((s) => (
            <SizeBadge>{s}</SizeBadge>
          ))}
        </div>
        <p class="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-muted">
          <span class="inline-flex items-center gap-1.5">
            <Icon name="tag" class="size-4" />
            {model.tags.length} Tags
          </span>
          {model.updated ? <Updated iso={model.updated} /> : null}
        </p>
      </a>
    </li>
  )
}
