/**
 * Static site generation entry point. scripts/build.mjs bundles this file with esbuild, calls
 * renderSite() and writes the returned files into dist/. Nothing here runs in production:
 * the deployed site is plain files (HTML, CSS, JS, JSON, text).
 *
 * URL → file mapping ("clean URLs"): "/" → index.html, "/docs/api" → docs/api.html,
 * "/library/laya:en" → library/laya:en.html. Cloudflare (html_handling: auto-trailing-slash),
 * GitHub Pages, Netlify and `serve` all resolve an extensionless URL to the .html file.
 *
 * Tag URLs contain a colon. Cloudflare's asset server percent-encodes it (307 to
 * /library/laya%3Aen), so each tag page is also written to library/<model>/tags/<tag>.html and
 * public/_redirects rewrites /library/<model>:<tag> to it with status 200 — the address bar keeps
 * the colon. Other static hosts serve the colon file directly.
 *
 * The /v2/ prefix is reserved for the static model registry; nothing here may write under it.
 */
import type { Child } from 'hono/jsx'
import { Layout, type PageMeta } from './components/Layout'
import { catalog, fullName, searchIndex } from './data/catalog'
import { docPages } from './generated/content'
import { hasAsset, setAssetVersions } from './lib/assets'
import { DocsIndex, DocView, withOrigin } from './pages/docs'
import { DownloadPage } from './pages/download'
import { HomePage } from './pages/home'
import { LibraryRedirect, ModelPage, TagPage, TagsPage } from './pages/library'
import { robotsTxt, sitemapXml } from './pages/meta'
import { NotFoundPage } from './pages/notFound'
import { SearchPage } from './pages/search'

export interface BuildOptions {
  /** Public origin without trailing slash, e.g. from the SITE_ORIGIN env var. */
  origin: string
  /** Content hashes of files in dist/static (for ?v= cache busting). */
  assetVersions: Record<string, string>
}

export interface OutputFile {
  /** Path relative to dist/. */
  path: string
  body: string
}

interface Page {
  url: string
  meta: PageMeta
  render: () => Child
  /** Extra copies of the page (paths relative to dist/). */
  aliases?: string[]
}

/** Every HTML page of the site. Also the sitemap, in this order. */
function pages(origin: string): Page[] {
  const list: Page[] = [
    { url: '/', meta: {}, render: () => <HomePage /> },
    {
      url: '/search',
      meta: {
        title: 'Models',
        description: 'Open decision models you can run locally with Ollaya.',
        nav: 'models',
        hideNavSearch: true,
      },
      render: () => <SearchPage />,
    },
    {
      url: '/download',
      meta: {
        title: 'Download',
        description: 'Install Ollaya on Linux or macOS with one command, or run the Docker image.',
        nav: 'download',
      },
      render: () => <DownloadPage origin={origin} />,
    },
    {
      url: '/docs',
      meta: {
        title: 'Docs',
        description: 'Ollaya documentation: quickstart, CLI, REST API, Modelfile and TypeSafe compatibility.',
        nav: 'docs',
      },
      render: () => <DocsIndex />,
    },
    ...docPages.map((page) => ({
      url: `/docs/${page.slug}`,
      meta: { title: page.title, description: page.description, nav: 'docs' as const },
      render: () => <DocView page={page} html={withOrigin(page.html, origin)} />,
    })),
  ]

  for (const model of catalog) {
    const card = `og/${model.name}.png`
    const ogImage = hasAsset(card) ? card : undefined
    list.push(
      {
        url: `/library/${model.name}`,
        meta: { title: model.name, description: model.description, nav: 'models', ogImage },
        render: () => <ModelPage model={model} />,
      },
      {
        url: `/library/${model.name}/tags`,
        meta: {
          title: `${model.name} tags`,
          description: `All tags of ${model.name}, including fp16 and fp32 precision variants.`,
          nav: 'models',
          ogImage,
        },
        render: () => <TagsPage model={model} />,
      },
      ...model.tags.map((tag) => ({
        url: `/library/${fullName(model, tag)}`,
        meta: {
          title: fullName(model, tag),
          description: `${fullName(model, tag)} — ${tag.summary}`,
          nav: 'models' as const,
          ogImage,
        },
        render: () => <TagPage model={model} tag={tag} />,
        // Target of the Cloudflare _redirects rewrite for colon URLs (see the header comment).
        aliases: [`library/${model.name}/tags/${tag.name}.html`],
      })),
    )
  }
  return list
}

/** URL path → file in dist/. */
export function fileForUrl(url: string): string {
  return url === '/' ? 'index.html' : `${url.slice(1)}.html`
}

async function toHtml(node: Child): Promise<string> {
  return `<!DOCTYPE html>${await (await (node as Promise<string> | string)).toString()}`
}

export async function renderSite({ origin, assetVersions }: BuildOptions): Promise<OutputFile[]> {
  setAssetVersions(assetVersions)
  const out: OutputFile[] = []
  const all = pages(origin)

  for (const page of all) {
    const html = await toHtml(
      <Layout origin={origin} path={page.url} meta={page.meta}>
        {page.render()}
      </Layout>,
    )
    out.push({ path: fileForUrl(page.url), body: html })
    for (const alias of page.aliases ?? []) out.push({ path: alias, body: html })
  }

  out.push({
    path: '404.html',
    body: await toHtml(
      <Layout origin={origin} path="/404" meta={{ title: 'Page not found', noindex: true }}>
        <NotFoundPage />
      </Layout>,
    ),
  })
  out.push({ path: 'library.html', body: await toHtml(<LibraryRedirect origin={origin} />) })
  out.push({ path: 'robots.txt', body: robotsTxt(origin) })
  out.push({ path: 'sitemap.xml', body: sitemapXml(origin, all.map((p) => p.url)) })
  out.push({ path: 'search.json', body: `${JSON.stringify(searchIndex())}\n` })

  const seen = new Set<string>()
  for (const f of out) {
    if (f.path.startsWith('v2/')) throw new Error(`${f.path}: /v2/ is reserved for the model registry`)
    if (seen.has(f.path)) throw new Error(`${f.path}: generated twice`)
    seen.add(f.path)
  }
  return out
}
