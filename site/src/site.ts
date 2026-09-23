/** Site-wide constants. The public host is NOT here on purpose — see env.ts / SITE_ORIGIN. */
export const SITE_NAME = 'Ollaya'
export const TAGLINE = 'Run decision models locally.'
export const SITE_DESCRIPTION =
  'Ollaya downloads and serves open decision models on your own machine. Typed, calibrated answers in milliseconds — private and open source.'

export const GITHUB_URL = 'https://github.com/cobanov/ollaya'
export const DOCKER_IMAGE = 'ghcr.io/cobanov/ollaya'

/** Default address of the local Ollaya server (the runtime, not this website). */
export const LOCAL_PORT = 11435
export const LOCAL_API = `http://localhost:${LOCAL_PORT}`

export const COPYRIGHT_YEAR = 2026
