import type { Model } from '../data/catalog'
import { AUTHOR, GITHUB_URL, HF_URL, SITE_DESCRIPTION, SITE_NAME } from '../site'

/** /v2/ (manifests) and /blobs/ belong to the static model registry, which is not for crawlers. */
export function robotsTxt(origin: string): string {
  return `User-agent: *\nAllow: /\nDisallow: /v2/\nDisallow: /blobs/\n\nSitemap: ${origin}/sitemap.xml\n`
}

/** schema.org SoftwareApplication for the home page, so search engines and AI answers can cite what Ollaya is. */
export function softwareJsonLd(origin: string): object {
  return {
    '@context': 'https://schema.org',
    '@type': 'SoftwareApplication',
    name: SITE_NAME,
    description: SITE_DESCRIPTION,
    url: origin,
    applicationCategory: 'DeveloperApplication',
    operatingSystem: 'macOS, Windows, Linux',
    downloadUrl: `${origin}/download`,
    softwareHelp: `${origin}/docs`,
    license: 'https://www.apache.org/licenses/LICENSE-2.0',
    offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
    keywords: 'decision models, Jev-compatible API, self-hosted, /v1/systemone, TypeSafe SDK, MCP server, local inference',
    author: { '@type': 'Person', name: AUTHOR.name, url: AUTHOR.url },
    sameAs: [GITHUB_URL, HF_URL],
  }
}

/** schema.org for a model page: the model with its author, license and source, and its breadcrumb under Models. */
export function modelJsonLd(model: Model, origin: string): object {
  const url = `${origin}/library/${model.name}`
  // A license id (Apache-2.0, MIT) has an SPDX page; a free-text license stays on the page only.
  const license = /^[\w.+-]+$/.test(model.license) ? `https://spdx.org/licenses/${model.license}` : undefined
  return {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'SoftwareApplication',
        name: model.name,
        alternateName: model.title !== model.name ? model.title : undefined,
        description: model.description,
        url,
        applicationCategory: 'DeveloperApplication',
        applicationSubCategory: 'Decision model',
        operatingSystem: 'macOS, Windows, Linux',
        softwareRequirements: `${SITE_NAME}, ${origin}/download`,
        isAccessibleForFree: true,
        author: model.publisher ?? undefined,
        license,
        isBasedOn: model.source ?? undefined,
        dateModified: model.updated ?? undefined,
        keywords: model.keywords.join(', '),
      },
      {
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Models', item: `${origin}/search` },
          { '@type': 'ListItem', position: 2, name: model.name, item: url },
        ],
      },
    ],
  }
}

const text = (html: string) =>
  html
    .replace(/<[^>]+>/g, '')
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"')
    .replaceAll('&#39;', "'")
    .replaceAll('&amp;', '&')
    .replace(/\s+/g, ' ')
    .trim()

/** schema.org FAQPage from a rendered FAQ: each <h2> is a question, everything up to the next <h2> its answer. */
export function faqJsonLd(html: string): object {
  const qa = html
    .split(/<h2[^>]*>/)
    .slice(1)
    .map((part) => {
      const [q = '', a = ''] = part.split('</h2>')
      return { '@type': 'Question', name: text(q), acceptedAnswer: { '@type': 'Answer', text: text(a) } }
    })
  if (qa.length < 5 || qa.some((x) => !x.name.endsWith('?') || !x.acceptedAnswer.text))
    throw new Error('faqJsonLd: expected a FAQ of "## Question?" sections, each with an answer')
  return { '@context': 'https://schema.org', '@type': 'FAQPage', mainEntity: qa }
}

export function sitemapXml(origin: string, paths: string[]): string {
  const urls = paths.map((p) => `  <url><loc>${origin}${p}</loc></url>`).join('\n')
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`
}
