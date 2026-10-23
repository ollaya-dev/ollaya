import { catalog } from '../data/catalog'
import { docPages } from '../generated/content'
import { hasAsset } from '../lib/assets'
import { benchUrl } from '../lib/bench'
import { AUTHOR, COPYRIGHT_YEAR, GITHUB_URL, HF_URL, ISSUES_URL } from '../site'
import { LogoMark } from './Logo'

/** The footer's link groups: every page the site has, by what the visitor is after. */
const docNav = (slug: string) => {
  const d = docPages.find((p) => p.slug === slug)
  if (!d) throw new Error(`footer: no doc page ${slug}`)
  return { href: `/docs/${slug}`, label: d.nav }
}
const footerGroups: { title: string; links: { href: string; label: string }[] }[] = [
  {
    title: 'Product',
    links: [
      { href: '/download', label: 'Download' },
      { href: '/search', label: 'Models' },
      ...(hasAsset('try.js') ? [{ href: '/try', label: 'Try in the browser' }] : []),
      { href: benchUrl(), label: 'Ollaya Bench' },
      { href: '/stats', label: 'Stats' },
      { href: '/blog', label: 'Blog' },
    ],
  },
  { title: 'Docs', links: ['quickstart', 'cli', 'api', 'modelfile', 'typesafe-compatibility', 'faq'].map(docNav) },
  { title: 'Use cases', links: ['agents', 'guardrails', 'routing', 'replace-llm-classifiers', 'cpu-classification', 'integrations'].map(docNav) },
  {
    title: 'Models',
    links: [...catalog]
      .sort((a, b) => a.rank - b.rank)
      .slice(0, 6)
      .map((m) => ({ href: `/library/${m.name}`, label: m.name })),
  },
  {
    title: 'Community',
    links: [
      { href: GITHUB_URL, label: 'GitHub' },
      { href: ISSUES_URL, label: 'Issues' },
      { href: HF_URL, label: 'Hugging Face' },
      { href: '/contributors', label: 'Contributors' },
      docNav('model-authors'),
      { href: '/llms.txt', label: 'llms.txt' },
    ],
  },
]

export function Footer() {
  return (
    <footer class="frame">
      <div class="rule flex flex-col gap-6 px-6 py-10 md:flex-row md:items-start md:justify-between">
        <div class="max-w-sm">
          <a href="/" class="inline-flex items-center gap-2 text-fg" aria-label="Ollaya home">
            <LogoMark class="size-8" />
            <span class="text-lg font-semibold tracking-tight">ollaya</span>
          </a>
          <p class="mt-4 text-[15px] text-body">Run open decision models locally. Typed questions in, calibrated answers out, in milliseconds, on your own hardware.</p>
        </div>
        <div class="flex items-center gap-2">
          <a href={GITHUB_URL} class="inline-flex size-10 items-center justify-center rounded-md border border-line text-body hover:border-line-strong hover:text-fg" aria-label="GitHub">
            <svg viewBox="0 0 24 24" class="size-5" fill="currentColor" aria-hidden="true">
              <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
            </svg>
          </a>
          <a href={HF_URL} class="inline-flex h-10 items-center gap-2 rounded-md border border-line px-3 text-sm text-body hover:border-line-strong hover:text-fg">
            Hugging Face
          </a>
          <a href="/download" class="inline-flex h-10 items-center rounded-md bg-btn px-4 text-sm font-medium text-btn-fg hover:bg-btn-hover">
            Download
          </a>
        </div>
      </div>
      <nav aria-label="Footer" class="rule grid grid-cols-2 gap-px bg-line md:grid-cols-5">
        {footerGroups.map((g, i) => (
          <div class={`bg-canvas px-6 py-8 ${i === footerGroups.length - 1 ? 'col-span-2 md:col-span-1' : ''}`}>
            <h3 class="text-sm font-semibold text-fg">{g.title}</h3>
            <ul class="mt-4 space-y-2.5 text-[15px]" role="list">
              {g.links.map((l) => (
                <li>
                  <a href={l.href} class={`text-body hover:text-fg ${g.title === 'Models' ? 'font-mono text-[14px]' : ''}`}>
                    {l.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
      <div class="rule flex flex-col gap-3 px-6 py-6 text-[13px] text-muted sm:flex-row sm:items-center sm:justify-between">
        <p>
          © {COPYRIGHT_YEAR} Ollaya · Apache-2.0 · Built by{' '}
          <a href={AUTHOR.url} class="text-body hover:text-fg">
            {AUTHOR.name}
          </a>
        </p>
        <p>An independent open-source project, not affiliated with Ollama or TypeSafe.</p>
      </div>
    </footer>
  )
}

