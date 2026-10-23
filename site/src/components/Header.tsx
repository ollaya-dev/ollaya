import { hasAsset } from '../lib/assets'
import { benchUrl } from '../lib/bench'
import { githubStars } from '../lib/stars'
import { GITHUB_URL } from '../site'
import { Icon } from './Icon'
import { LogoMark } from './Logo'

export type NavKey = 'models' | 'results' | 'docs' | 'download' | 'try'
/** What a page highlights in the top nav: a main section, or the blog, which is linked after them. */
export type PageNav = NavKey | 'blog'

const allLinks: { key: PageNav; href: string; label: string }[] = [
  { key: 'models', href: '/search', label: 'Models' },
  { key: 'try', href: '/try', label: 'Try' },
  // Ollaya Bench, on its own host: the href is set at render time (benchUrl needs the build's origin).
  { key: 'results', href: '', label: 'Bench' },
  { key: 'docs', href: '/docs', label: 'Docs' },
  { key: 'blog', href: '/blog', label: 'Blog' },
]
// /try exists only when the build bundled the in-browser runtime (scripts/build.mjs).
const navLinks = () => allLinks.filter((l) => l.key !== 'try' || hasAsset('try.js'))
const hrefOf = (l: (typeof allLinks)[number]) => (l.key === 'results' ? benchUrl() : l.href)

function NavSearch({ id, big }: { id: string; big?: boolean }) {
  return (
    <form action="/search" method="get" role="search" class="relative w-full" data-typeahead={big ? undefined : true}>
      <label for={id} class="sr-only">
        Search models
      </label>
      <span class={`pointer-events-none absolute inset-y-0 left-3 flex items-center text-muted`}>
        <Icon name="search" class="size-4" />
      </span>
      <input
        id={id}
        name="q"
        type="search"
        placeholder="Search models"
        autocomplete="off"
        spellcheck={false}
        class={`peer w-full rounded-md border border-line bg-canvas/60 pr-12 pl-9 text-sm text-fg hover:border-line-strong focus:border-line-strong focus:ring-2 focus:ring-fill-strong focus:outline-none [&::-webkit-search-cancel-button]:hidden ${big ? 'h-11 text-base' : 'h-9'}`}
        aria-controls={big ? undefined : 'nav-typeahead'}
      />
      {big ? null : (
        <kbd class="pointer-events-none absolute inset-y-0 right-2.5 my-auto flex h-5 items-center rounded border border-line px-1.5 font-sans text-[11px] text-muted peer-focus:hidden" aria-hidden="true" data-shortcut-hint>
          ⌘K
        </kbd>
      )}
      <button type="submit" class="sr-only">
        Search
      </button>
      {big ? null : <div id="nav-typeahead" class="panel panel-strong absolute top-full right-0 left-0 z-50 mt-2 overflow-hidden rounded-md" hidden></div>}
    </form>
  )
}

export function Header({ active, hideSearch }: { active?: PageNav; hideSearch?: boolean }) {
  const stars = githubStars()
  return (
    <header class="fixed inset-x-0 top-0 z-40 border-b border-line bg-canvas/70 backdrop-blur-xl">
      <nav aria-label="Main" class="mx-auto flex h-14 max-w-6xl items-center gap-6 px-5">
        <a href="/" class="flex shrink-0 items-center gap-2 text-fg" aria-label="Ollaya home">
          <LogoMark class="size-7" />
          <span class="text-[17px] font-semibold tracking-[-0.035em]">ollaya</span>
        </a>
        <ul class="hidden items-center gap-5 text-sm md:flex">
          {navLinks().map((l) => (
            <li>
              <a href={hrefOf(l)} class={active === l.key ? 'font-medium text-fg' : 'text-body hover:text-fg'} aria-current={active === l.key ? 'page' : undefined}>
                {l.label}
              </a>
            </li>
          ))}
        </ul>
        <div class="hidden min-w-0 flex-1 justify-end lg:flex">{hideSearch ? null : <div class="w-64"><NavSearch id="nav-search" /></div>}</div>
        <div class="ml-auto flex items-center gap-2 lg:ml-0">
          <a href={GITHUB_URL} class="hidden h-9 items-center gap-1.5 rounded-md border border-line px-3 text-sm text-body hover:border-line-strong hover:text-fg sm:inline-flex">
            <Icon name="star" class="size-3.5" />
            {stars ?? 'GitHub'}
            {stars ? <span class="sr-only"> stars</span> : null}
          </a>
          <a href="/download" class="inline-flex h-9 items-center rounded-md bg-btn px-4 text-sm font-medium text-btn-fg hover:bg-btn-hover" aria-current={active === 'download' ? 'page' : undefined}>
            Download
          </a>
          <details class="group md:hidden" data-mobile-menu>
            <summary class="inline-flex size-9 cursor-pointer items-center justify-center rounded-md border border-line text-fg" aria-label="Menu">
              <Icon name="menu" class="size-5 group-open:hidden" />
              <Icon name="close" class="hidden size-5 group-open:block" />
            </summary>
            <div class="fixed inset-x-0 top-14 bottom-0 z-50 overflow-y-auto bg-canvas px-5 pt-4 pb-12">
              <NavSearch id="mobile-search" big />
              <ul class="mt-6 flex flex-col gap-1 text-2xl font-medium text-fg">
                {[...navLinks().map((l) => [hrefOf(l), l.label] as const), [GITHUB_URL, 'GitHub'] as const, ['/download', 'Download'] as const].map(([href, label]) => (
                  <li>
                    <a href={href} class="block py-2 hover:underline underline-offset-4">
                      {label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </details>
        </div>
      </nav>
    </header>
  )
}
