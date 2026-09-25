/** Site-wide constants. The public host is NOT here on purpose — see SITE_ORIGIN in scripts/build.mjs. */
export const SITE_NAME = 'Ollaya'
export const TAGLINE = 'Run decision models locally.'
export const SITE_DESCRIPTION =
  'Ollaya downloads and serves open decision models on your own machine. Typed, calibrated answers in milliseconds — private and open source.'

export const GITHUB_URL = 'https://github.com/ollaya-dev/ollaya'
export const RELEASES_URL = `${GITHUB_URL}/releases`
/** The latest release's assets by name (the desktop installers have stable names). */
export const LATEST_DOWNLOAD = `${RELEASES_URL}/latest/download`
/** False until the Windows installer is code-signed: the download page then says how to get past SmartScreen. */
export const WINDOWS_APP_SIGNED = false
export const ISSUES_URL = `${GITHUB_URL}/issues`
/** Derived files (ONNX graphs, configs) of every model, mirrored on Hugging Face. No weights. */
export const HF_URL = 'https://huggingface.co/ollaya-dev'
/** Planned features, tracked on GitHub. */
export const MCP_ISSUE_URL = `${GITHUB_URL}/issues/1`
export const SKILL_ISSUE_URL = `${GITHUB_URL}/issues/2`

/** Container images: CPU (linux/amd64, linux/arm64) and NVIDIA GPU (`:cuda`, linux/amd64). */
export const DOCKER_IMAGE = 'ghcr.io/ollaya-dev/ollaya'

/** Default address of the local Ollaya server (the runtime, not this website). */
export const LOCAL_PORT = 11435
export const LOCAL_API = `http://localhost:${LOCAL_PORT}`

export const COPYRIGHT_YEAR = 2026

/**
 * Visit counting with Open Analytics (open source, cookieless, no personal data), self-hosted at
 * oa.cobanov.run. `key` is the site's public tracking key (oa_pk_…); empty leaves the script out.
 * Its host must also be allowed in public/_headers (script-src and connect-src).
 */
export const ANALYTICS = {
  src: 'https://oa-c.cobanov.run/oa.js',
  collector: 'https://oa-c.cobanov.run',
  key: 'oa_pk_V20O3SgpVMTojY7jWQ0x1i_QtXXneyRW',
}
