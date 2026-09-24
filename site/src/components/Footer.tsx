import { COPYRIGHT_YEAR, GITHUB_URL, HF_URL } from '../site'

const links = [
  { href: '/download', label: 'Download' },
  { href: '/docs', label: 'Docs' },
  { href: GITHUB_URL, label: 'GitHub' },
  { href: HF_URL, label: 'Hugging Face' },
  { href: '/search', label: 'Models' },
]

export function Footer() {
  return (
    <footer class="mt-24 px-4 py-8 md:px-6">
      <div class="flex flex-col-reverse gap-4 text-xs text-muted sm:flex-row sm:items-center sm:justify-between">
        <p>© {COPYRIGHT_YEAR} Ollaya</p>
        <nav aria-label="Footer">
          <ul class="flex flex-wrap gap-x-5 gap-y-2">
            {links.map((l) => (
              <li>
                <a href={l.href} class="underline-offset-4 hover:text-fg hover:underline">
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
