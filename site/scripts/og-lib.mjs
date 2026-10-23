// Shared pieces of the social preview cards (1200×630 PNG): ollaya.dev's dark card style, Geist
// inlined from Google Fonts, and rendering through a headless Chromium. Used by og-images.mjs (the
// committed cards) and og-bench.mjs (the bench's per-page cards, made from an edition at deploy time).

import { spawnSync } from 'node:child_process'
import { existsSync, readdirSync } from 'node:fs'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { homedir, tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

export const OWL = `<svg viewBox="0 0 32 32"><circle cx="9.5" cy="16" r="6.5" fill="none" stroke="currentColor" stroke-width="3.2"/><circle cx="22.5" cy="16" r="6.5" fill="none" stroke="currentColor" stroke-width="3.2"/><circle cx="22.5" cy="16" r="2.6" fill="currentColor"/></svg>`

export const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')

const FONTS_CSS = 'https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=block'

/** Geist and Geist Mono as @font-face rules with the files inlined, so the screenshot never races the network. */
export async function inlineFonts() {
  try {
    const ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36'
    let css = await (await fetch(FONTS_CSS, { headers: { 'user-agent': ua } })).text()
    for (const url of new Set(css.match(/https:\/\/fonts\.gstatic\.com\/[^)]+/g) ?? [])) {
      const font = Buffer.from(await (await fetch(url)).arrayBuffer()).toString('base64')
      css = css.replaceAll(url, `data:font/woff2;base64,${font}`)
    }
    return css
  } catch (e) {
    console.warn(`og: could not load Geist (${e.message}); using system fonts`)
    return ''
  }
}

/** A card page: the base style, `css` for the card's own classes, and its body. */
export function page(fonts, body, css = '') {
  return `<!doctype html><html><head><meta charset="utf-8">
<style>${fonts}</style>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { width: 1200px; height: 630px; overflow: hidden; background: #0a0a0a; color: #fafafa;
    font-family: Geist, system-ui, sans-serif; -webkit-font-smoothing: antialiased; }
  .card { position: relative; width: 1200px; height: 630px; padding: 64px 72px; display: flex; flex-direction: column; }
  .glow { position: absolute; inset: 0; pointer-events: none;
    background: radial-gradient(640px 360px at 100% 0%, rgba(121,192,255,.13), transparent 70%),
                radial-gradient(520px 300px at 0% 100%, rgba(126,226,168,.08), transparent 70%); }
  .grid { position: absolute; inset: 0; pointer-events: none; opacity: .5;
    background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
    background-size: 48px 48px; mask-image: radial-gradient(900px 500px at 80% 10%, #000 20%, transparent 75%); }
  .top { position: relative; display: flex; align-items: center; justify-content: space-between; }
  .brand { display: flex; align-items: center; gap: 16px; font-size: 38px; font-weight: 500; letter-spacing: -.02em; }
  .brand svg { width: 48px; height: 48px; }
  .url { font-family: 'Geist Mono', ui-monospace, monospace; font-size: 22px; color: #8a8a8a; }
  .main { position: relative; margin-top: auto; }
  .badge { display: inline-block; font-size: 18px; font-weight: 500; letter-spacing: .08em; text-transform: uppercase;
    color: #79c0ff; border: 1px solid rgba(121,192,255,.35); background: rgba(121,192,255,.08); border-radius: 999px; padding: 6px 16px; }
  h1 { font-size: 84px; line-height: 1.02; font-weight: 500; letter-spacing: -.035em; }
  h1.model { font-family: 'Geist Mono', ui-monospace, monospace; font-size: 112px; letter-spacing: -.04em; margin-top: 22px; }
  .text { margin-top: 22px; max-width: 980px; font-size: 32px; line-height: 1.35; color: #a3a3a3; letter-spacing: -.01em; }
  .bottom { position: relative; margin-top: 44px; display: flex; align-items: center; justify-content: space-between; }
  .cmd { font-family: 'Geist Mono', ui-monospace, monospace; font-size: 26px; background: #141414; border: 1px solid #262626;
    border-radius: 14px; padding: 16px 24px; box-shadow: inset 0 1px 0 rgba(255,255,255,.05); }
  .p { color: #7ee2a8; } .c { color: #ffc36b; }
  .meta { font-size: 22px; color: #8a8a8a; text-align: right; line-height: 1.5; }
  .meta b { color: #d4d4d4; font-weight: 500; }
${css}
</style></head><body><div class="card"><div class="glow"></div><div class="grid"></div>${body}</div></body></html>`
}

export const top = (url, name = 'ollaya') =>
  `<div class="top"><div class="brand">${OWL}<span>${name}</span></div><div class="url">${esc(url)}</div></div>`

export function chrome() {
  if (process.env.CHROME) return process.env.CHROME
  const cache = join(homedir(), '.cache', 'ms-playwright')
  const dirs = existsSync(cache) ? readdirSync(cache).filter((d) => d.startsWith('chromium_headless_shell-')).sort() : []
  for (const d of dirs.reverse()) {
    for (const rel of ['chrome-headless-shell-linux64/chrome-headless-shell', 'chrome-headless-shell-mac-arm64/chrome-headless-shell']) {
      const p = join(cache, d, rel)
      if (existsSync(p)) return p
    }
  }
  throw new Error('no Chromium found: set CHROME to a Chrome or chrome-headless-shell binary')
}

/** Renders each card ({ file, html, dir? }) to <dir or out>/<file> with Chromium. */
export async function renderCards(cards, out) {
  const bin = chrome()
  const tmp = await mkdtemp(join(tmpdir(), 'ollaya-og-'))
  try {
    for (const card of cards) {
      const html = join(tmp, `${card.file.replaceAll('/', '-')}.html`)
      await writeFile(html, card.html)
      const target = join(card.dir ?? out, card.file)
      await mkdir(dirname(target), { recursive: true })
      const r = spawnSync(
        bin,
        ['--no-sandbox', '--hide-scrollbars', '--force-device-scale-factor=1', '--window-size=1200,630', '--virtual-time-budget=5000', `--screenshot=${target}`, `file://${html}`],
        { encoding: 'utf8' },
      )
      if (r.status !== 0 || !existsSync(target)) throw new Error(`rendering ${card.file} failed:\n${r.stderr}`)
      console.log(`og: ${card.file}`)
    }
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}
