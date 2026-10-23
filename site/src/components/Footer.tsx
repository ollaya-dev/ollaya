import { benchUrl } from '../lib/bench'
import { AUTHOR, COPYRIGHT_YEAR, GITHUB_URL, HF_URL } from '../site'

// Six links: two even rows of three on a phone, one row from sm up.
const links = [
  { href: '/search', label: 'Models' },
  { href: '', label: 'Bench' },
  { href: '/docs', label: 'Docs' },
  { href: '/download', label: 'Download' },
  { href: '/contributors', label: 'Contributors' },
  { href: GITHUB_URL, label: 'GitHub' },
  { href: HF_URL, label: 'Hugging Face' },
]

export function Footer() {
  return (
    <footer class="mt-24 border-t border-line">
      <div class="mx-auto flex max-w-6xl flex-col-reverse gap-6 px-4 py-8 text-xs text-muted sm:flex-row sm:items-center sm:justify-between sm:gap-4 md:px-6">
        <p class="border-t border-line pt-6 sm:border-0 sm:pt-0">
          © {COPYRIGHT_YEAR} Ollaya. Built by{' '}
          <a href={AUTHOR.url} class="text-body underline-offset-4 hover:text-fg hover:underline">
            {AUTHOR.name}
          </a>
          .
        </p>
        <nav aria-label="Footer">
          <ul class="grid grid-cols-3 gap-x-4 gap-y-3 sm:flex sm:flex-wrap sm:gap-x-5 sm:gap-y-2">
            {links.map((l) => (
              <li>
                <a href={l.label === 'Bench' ? benchUrl() : l.href} class="underline-offset-4 hover:text-fg hover:underline">
                  {l.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </footer>
  )
}
