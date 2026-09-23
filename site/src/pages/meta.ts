import { GITHUB_URL } from '../site'

/**
 * Placeholder installer served at /install.sh. `curl -fsSL <origin>/install.sh | sh` prints a
 * notice and exits 1 until the first release exists. Replace with the real installer at release.
 */
export function installScript(): string {
  return `#!/bin/sh
# Ollaya installer
#
# Ollaya has not been released yet. This placeholder only prints a notice and
# exits with an error, so scripts that pipe it to sh fail loudly instead of
# silently doing nothing.
#
# Source code and release updates: ${GITHUB_URL}

set -eu

cat >&2 <<'NOTICE'

  Ollaya has not been released yet.

  The first release is in progress. Star or watch the repository
  to hear when it ships:

    ${GITHUB_URL}

NOTICE

exit 1
`
}

/** /v2/ is reserved for the static model registry (manifests), which is not for crawlers. */
export function robotsTxt(origin: string): string {
  return `User-agent: *\nAllow: /\nDisallow: /v2/\n\nSitemap: ${origin}/sitemap.xml\n`
}

export function sitemapXml(origin: string, paths: string[]): string {
  const urls = paths.map((p) => `  <url><loc>${origin}${p}</loc></url>`).join('\n')
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`
}
