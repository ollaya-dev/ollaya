import { GITHUB_URL } from '../site'
import { Icon } from './Icon'
import { Logo } from './Logo'

export type NavKey = 'models' | 'docs' | 'download'

const links: { key: NavKey | 'github'; href: string; label: string }[] = [
  { key: 'models', href: '/search', label: 'Models' },
  { key: 'docs', href: '/docs', label: 'Docs' },
  { key: 'github', href: GITHUB_URL, label: 'GitHub' },
]

function NavSearch() {
  return (
    <form action="/search" method="get" role="search" class="relative w-full" data-typeahead>
      <label for="nav-search" class="sr-only">
        Search models
      </label>
      <span class="pointer-events-none absolute inset-y-0 left-3.5 flex items-center text-muted">
        <Icon name="search" class="size-4" />
      </span>
      <input
        id="nav-search"
        name="q"
        type="search"
        placeholder="Search models"
        autocomplete="off"
        spellcheck={false}
        class="peer w-full rounded-full border border-line bg-canvas py-2 pr-14 pl-10 text-sm text-fg hover:border-line-strong focus:border-line-strong focus:ring-4 focus:ring-fill-strong focus:outline-none [&::-webkit-search-cancel-button]:hidden"
        aria-controls="nav-typeahead"
      />
      <kbd
        class="pointer-events-none absolute inset-y-0 right-3 my-auto flex h-5 items-center rounded border border-line px-1.5 font-sans text-[11px] text-muted peer-focus:hidden"
        aria-hidden="true"
        data-shortcut-hint
      >
        ⌘K
      </kbd>
      <button type="submit" class="sr-only">
        Search
      </button>
      <div
        id="nav-typeahead"
        class="absolute top-full right-0 left-0 z-50 mt-2 overflow-hidden rounded-2xl border border-line bg-canvas shadow-2xl shadow-black/5"
        hidden
      ></div>
    </form>
  )
}

function MobileMenu() {
  return (
    <details class="group md:hidden" data-mobile-menu>
      <summary
        class="inline-flex size-10 cursor-pointer items-center justify-center rounded-full text-fg hover:bg-fill"
        aria-label="Menu"
      >
        <Icon name="menu" class="size-6 group-open:hidden" />
        <Icon name="close" class="hidden size-6 group-open:block" />
      </summary>
      <div class="fixed inset-x-0 top-16 bottom-0 z-50 overflow-y-auto bg-canvas px-4 pt-4 pb-12">
        <form action="/search" method="get" role="search" class="relative">
          <label for="mobile-search" class="sr-only">
            Search models
          </label>
          <span class="pointer-events-none absolute inset-y-0 left-4 flex items-center text-muted">
            <Icon name="search" class="size-5" />
          </span>
          <input
            id="mobile-search"
            name="q"
            type="search"
            placeholder="Search models"
            autocomplete="off"
            class="w-full rounded-full border border-line bg-canvas py-3 pr-4 pl-12 text-base text-fg focus:border-line-strong focus:ring-4 focus:ring-fill-strong focus:outline-none"
          />
          <button type="submit" class="sr-only">
            Search
          </button>
        </form>
        <ul class="mt-8 flex flex-col gap-1 text-2xl font-medium text-fg">
          {links.map((l) => (
            <li>
              <a href={l.href} class="block py-2 hover:underline underline-offset-4">
                {l.label}
              </a>
            </li>
          ))}
          <li>
            <a href="/download" class="block py-2 hover:underline underline-offset-4">
              Download
            </a>
          </li>
        </ul>
      </div>
    </details>
  )
}

export function Header({ active, hideSearch }: { active?: NavKey; hideSearch?: boolean }) {
  return (
    <header class="sticky top-0 z-40 bg-canvas">
      <nav aria-label="Main" class="flex h-16 w-full items-center gap-4 px-4 md:gap-6 md:px-6">
        <div class="flex items-baseline gap-6 lg:flex-1 lg:basis-0">
          <a href="/" class="flex rounded-md" aria-label="Ollaya home">
            <Logo />
          </a>
          <ul class="hidden items-baseline gap-6 text-sm md:flex">
            {links.map((l) => (
              <li>
                <a
                  href={l.href}
                  class={`underline-offset-4 hover:text-fg hover:underline ${active === l.key ? 'font-medium text-fg' : 'text-body'}`}
                  aria-current={active === l.key ? 'page' : undefined}
                >
                  {l.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
        <div class="hidden min-w-0 flex-1 justify-center md:flex lg:w-[22rem] lg:flex-none">
          {hideSearch ? null : <NavSearch />}
        </div>
        <div class="flex flex-1 items-center justify-end gap-2 md:flex-none lg:flex-1 lg:basis-0">
          <a
            href="/download"
            class="hidden items-center rounded-full bg-btn px-4 py-2 text-sm font-medium text-btn-fg hover:bg-btn-hover md:inline-flex"
            aria-current={active === 'download' ? 'page' : undefined}
          >
            Download
          </a>
          <MobileMenu />
        </div>
      </nav>
    </header>
  )
}
