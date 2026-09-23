/** Content hashes of files in dist/static, set once by the static build before rendering. */
let versions: Record<string, string> = {}

export function setAssetVersions(v: Record<string, string>): void {
  versions = v
}

/** URL for a file in /static with a content-hash query for cache busting. */
export function asset(path: string): string {
  const v = versions[path]
  return v ? `/static/${path}?v=${v}` : `/static/${path}`
}
