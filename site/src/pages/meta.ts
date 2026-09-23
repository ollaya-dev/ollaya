/** /v2/ (manifests) and /blobs/ belong to the static model registry, which is not for crawlers. */
export function robotsTxt(origin: string): string {
  return `User-agent: *\nAllow: /\nDisallow: /v2/\nDisallow: /blobs/\n\nSitemap: ${origin}/sitemap.xml\n`
}

export function sitemapXml(origin: string, paths: string[]): string {
  const urls = paths.map((p) => `  <url><loc>${origin}${p}</loc></url>`).join('\n')
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`
}
