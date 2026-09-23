import type { Child } from 'hono/jsx'
import { Icon } from '../components/Icon'
import { docPages, type DocPage } from '../generated/content'

const escapeHtml = (s: string) =>
  s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;')

/**
 * Docs HTML is rendered from Markdown at build time; this fills in the {{SITE_ORIGIN}}
 * (https://host) and {{SITE_HOST}} (host) placeholders.
 */
export const withOrigin = (html: string, origin: string) =>
  html.replaceAll('{{SITE_ORIGIN}}', escapeHtml(origin)).replaceAll('{{SITE_HOST}}', escapeHtml(origin.replace(/^https?:\/\//, '')))

function NavList({ current }: { current?: string }) {
  return (
    <ul class="space-y-0.5 text-sm" role="list">
      <li>
        <a
          href="/docs"
          class={`block rounded-md px-3 py-1.5 hover:bg-fill hover:text-fg ${current === undefined ? 'bg-fill font-medium text-fg' : 'text-body'}`}
          aria-current={current === undefined ? 'page' : undefined}
        >
          Overview
        </a>
      </li>
      {docPages.map((p) => (
        <li>
          <a
            href={`/docs/${p.slug}`}
            class={`block rounded-md px-3 py-1.5 hover:bg-fill hover:text-fg ${current === p.slug ? 'bg-fill font-medium text-fg' : 'text-body'}`}
            aria-current={current === p.slug ? 'page' : undefined}
          >
            {p.nav}
          </a>
        </li>
      ))}
    </ul>
  )
}

function DocsShell({ current, title, children }: { current?: string; title: string; children: Child }) {
  return (
    <div class="mx-auto max-w-6xl px-4 pt-8 md:grid md:grid-cols-[12rem_minmax(0,1fr)] md:gap-12 md:px-6 md:pt-12 lg:gap-16">
      <aside>
        <details class="group mb-8 rounded-lg border border-line md:hidden">
          <summary class="flex cursor-pointer items-center justify-between px-4 py-3 text-sm font-medium text-fg">
            <span>
              <span class="text-muted">Docs · </span>
              {title}
            </span>
            <Icon name="chevronDown" class="size-4 text-muted transition-transform group-open:rotate-180" />
          </summary>
          <nav aria-label="Documentation menu" class="border-t border-line p-2">
            <NavList current={current} />
          </nav>
        </details>
        <nav aria-label="Documentation" class="sticky top-24 hidden md:block">
          <p class="px-3 pb-2 text-[13px] font-medium text-muted">Documentation</p>
          <NavList current={current} />
        </nav>
      </aside>
      <div class="min-w-0 max-w-3xl">{children}</div>
    </div>
  )
}

export function DocsIndex() {
  return (
    <DocsShell title="Overview">
      <h1 class="text-[30px] leading-tight font-medium tracking-tight text-fg">Documentation</h1>
      <p class="mt-3 text-lg text-body">
        Ollaya downloads and serves open decision models locally. A decision model reads a state — text, an email, a
        ticket, JSON — plus typed questions, and returns typed answers with calibrated probabilities in a single
        forward pass. It never generates text.
      </p>
      <ul class="mt-10 grid gap-4 sm:grid-cols-2" role="list">
        {docPages.map((p) => (
          <li>
            <a href={`/docs/${p.slug}`} class="group block h-full rounded-lg border border-line p-5 hover:border-line-strong">
              <span class="flex items-center justify-between text-sm font-semibold text-fg">
                {p.title}
                <Icon name="arrowRight" class="size-4 text-muted transition-transform group-hover:translate-x-0.5 group-hover:text-fg" />
              </span>
              <span class="mt-1.5 block text-[15px] text-body">{p.description}</span>
            </a>
          </li>
        ))}
      </ul>
    </DocsShell>
  )
}

export function DocView({ page, html }: { page: DocPage; html: string }) {
  const i = docPages.indexOf(page)
  const prev = docPages[i - 1]
  const next = docPages[i + 1]
  return (
    <DocsShell current={page.slug} title={page.nav}>
      <article class="prose" dangerouslySetInnerHTML={{ __html: html }}></article>
      <nav aria-label="Previous and next" class="mt-16 grid gap-4 border-t border-line pt-8 sm:grid-cols-2">
        {prev ? (
          <a href={`/docs/${prev.slug}`} class="group rounded-lg border border-line p-4 hover:border-line-strong">
            <span class="block text-[13px] text-muted">Previous</span>
            <span class="mt-0.5 block text-sm font-medium text-fg">{prev.title}</span>
          </a>
        ) : (
          <span class="hidden sm:block"></span>
        )}
        {next ? (
          <a href={`/docs/${next.slug}`} class="group rounded-lg border border-line p-4 text-right hover:border-line-strong">
            <span class="block text-[13px] text-muted">Next</span>
            <span class="mt-0.5 block text-sm font-medium text-fg">{next.title}</span>
          </a>
        ) : null}
      </nav>
    </DocsShell>
  )
}
