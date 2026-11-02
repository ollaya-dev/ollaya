/** The GitHub repository's star count, read once by the static build; undefined when offline. */
let stars: number | undefined

export function setGithubStars(n: number | undefined): void {
  stars = n
}

/** The exact count, for places that print it in full; undefined when the build could not read it. */
export function githubStarCount(): number | undefined {
  return stars
}

/** "495", "1.2k", "12k"; undefined when the build could not read the count. */
export function githubStars(): string | undefined {
  if (stars === undefined) return undefined
  if (stars < 1000) return String(stars)
  const k = stars / 1000
  return `${k < 10 ? k.toFixed(1).replace(/\.0$/, '') : Math.round(k)}k`
}
