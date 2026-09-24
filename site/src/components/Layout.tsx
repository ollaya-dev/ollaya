import type { Child } from 'hono/jsx'
import { asset } from '../lib/assets'
import { ANALYTICS, SITE_DESCRIPTION, SITE_NAME, TAGLINE } from '../site'
import { Footer } from './Footer'
import { Header, type NavKey } from './Header'

export interface PageMeta {
  /** Page title without the site suffix. Omit on the home page. */
  title?: string
  description?: string
  /** Highlighted top-nav item. */
  nav?: NavKey
  /** Hide the navbar search box (the /search page has its own). */
  hideNavSearch?: boolean
  noindex?: boolean
  /** Social card under /static (default og.png), e.g. "og/laya.png" for a model's pages. */
  ogImage?: string
}

export interface LayoutProps {
  /** Public origin without trailing slash, from the SITE_ORIGIN build variable. */
  origin: string
  /** URL path of this page, e.g. "/library/laya" (used for the canonical URL). */
  path: string
  meta: PageMeta
  children: Child
}

export function Layout({ origin, path, meta, children }: LayoutProps) {
  const canonical = `${origin}${path}`
  const title = meta.title ? `${meta.title} · ${SITE_NAME}` : `${SITE_NAME} — ${TAGLINE.replace(/\.$/, '')}`
  const description = meta.description ?? SITE_DESCRIPTION
  const ogImage = `${origin}${asset(meta.ogImage ?? 'og.png')}`

  return (
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{title}</title>
        <meta name="description" content={description} />
        {meta.noindex ? <meta name="robots" content="noindex" /> : <link rel="canonical" href={canonical} />}
        <meta name="color-scheme" content="light dark" />
        <meta name="theme-color" media="(prefers-color-scheme: light)" content="#ffffff" />
        <meta name="theme-color" media="(prefers-color-scheme: dark)" content="#0a0a0a" />
        <meta property="og:site_name" content={SITE_NAME} />
        <meta property="og:type" content="website" />
        <meta property="og:title" content={meta.title ?? `${SITE_NAME} — ${TAGLINE}`} />
        <meta property="og:description" content={description} />
        <meta property="og:url" content={canonical} />
        <meta property="og:image" content={ogImage} />
        <meta property="og:image:width" content="1200" />
        <meta property="og:image:height" content="630" />
        <meta property="og:image:alt" content={meta.ogImage && meta.title ? `${meta.title} on Ollaya` : 'Ollaya — run decision models locally'} />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:image" content={ogImage} />
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        {ANALYTICS.key ? (
          <script async src={ANALYTICS.src} data-key={ANALYTICS.key} data-collector={ANALYTICS.collector}></script>
        ) : null}
        <link rel="stylesheet" href={asset('app.css')} />
        <script src={asset('app.js')} defer></script>
      </head>
      <body class="flex min-h-screen flex-col bg-canvas font-sans text-body antialiased">
        <a
          href="#main"
          class="sr-only rounded-full bg-btn px-4 py-2 text-sm text-btn-fg focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-[60]"
        >
          Skip to content
        </a>
        <Header active={meta.nav} hideSearch={meta.hideNavSearch} />
        <main id="main" class="flex-1">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}
