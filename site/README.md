# Ollaya website

The website for Ollaya: home page, model library, docs, download page and `/install.sh`.
It is a **fully static site**. Pages are written as Hono JSX components, pre-rendered to HTML at
build time, and styled with Tailwind CSS v4. A small dependency-free script
(`public/static/app.js`) adds search, the navbar typeahead, tabs and copy buttons. There is no
server code and no Worker script, so hosting costs nothing.

## Develop

Requires Node.js 22 or newer.

```sh
npm ci
npm run build      # → dist/
npm run dev        # build, then serve dist/ with wrangler dev on http://localhost:8787
```

`wrangler dev` does not rebuild on its own. Run `npm run build` again after editing anything.

| Script | What it does |
|---|---|
| `npm run build` | Markdown → `src/generated/`, typecheck, pre-render everything into `dist/`, run checks |
| `npm run typecheck` | Generate content, then `tsc --noEmit` |
| `npm run dev` | `build` + `wrangler dev` (assets only, same routing as production) |
| `npm run deploy` | `build` + `wrangler deploy` |

The build fails if a host name is hard-coded in `src/`, `docs/`, `content/` or `public/`, if any
internal link in `dist/` points to a missing file, or if anything is written under `/v2/`.

## Layout

```
wrangler.jsonc          Assets-only Worker: ./dist, 404 page, custom domain route
scripts/gen-content.mjs Markdown (docs/, content/) → src/generated/content.ts (git-ignored)
scripts/build.mjs       dist/: copies public/, runs Tailwind, hashes assets, pre-renders pages
scripts/check.mjs       Post-build checks on dist/
src/build.tsx           Page list and URL → file mapping; renders every page, 404, sitemap, …
src/pages/              Page components: home, search, library, download, docs, 404, text files
src/components/         Layout, header, footer, logo, icons, code blocks, tabs, badges
src/data/catalog.ts     The model catalog — single source for search, model/tag pages, sitemap
src/data/examples.ts    Usage snippets (CLI / cURL / Python / JavaScript)
src/styles/app.css      Tailwind input: design tokens (light + dark) and prose styles
content/library/*.md    Model readmes
docs/*.md               Docs pages with front matter (title, nav, description, order)
public/                 Copied into dist/ as is: favicon, _headers, _redirects, static/app.js, images
```

`dist/` after a build:

```
index.html  search.html  download.html  docs.html  docs/<page>.html  404.html
library/laya.html  library/laya/tags.html  library/laya:<tag>.html  library/laya/tags/<tag>.html
library.html (meta refresh to /search)  search.json  install.sh  robots.txt  sitemap.xml
favicon.svg  static/{app.css, app.js, logo.svg, og.png}  _headers  _redirects
```

URLs have no `.html` suffix: `/docs/api` is served from `docs/api.html`.

### Search

`/search` lists every model in the HTML, so it works without JavaScript. `app.js` filters and
sorts the list in place and keeps `?q=&c=&o=` in the URL. The navbar typeahead loads
`/search.json`, which is generated from `src/data/catalog.ts`.

### Editing the catalog

Add or change models in `src/data/catalog.ts`; the readme goes in `content/library/<name>.md`.
Do not add pull counts or other numbers we do not have. Sizes are approximate until artifacts are
published.

## Hosting

The site is built for **Cloudflare Workers static assets**, with no `main` script. That means
`wrangler.jsonc` has only an `assets` block (`not_found_handling: "404-page"`,
`html_handling: "auto-trailing-slash"`), so no request runs any code.

`dist/` also works **unchanged on any static host**, for example GitHub Pages, Netlify or
`npx serve dist`. Two files are Cloudflare extras that other hosts ignore:

- `_headers`: security headers, `Content-Type` for `install.sh`, long cache for `/static/*`
  (files there are referenced with a `?v=<content hash>` query).
- `_redirects`:
  - a 302 from `/library` to `/search`. Other hosts serve `library.html`, which forwards with a
    meta refresh.
  - a 200 rewrite from `/library/<model>:<tag>` to `library/<model>/tags/<tag>.html`. Cloudflare's
    asset server would otherwise percent-encode the colon (307 to `/library/laya%3Aen`). Other
    hosts serve `library/laya:en.html` directly.

`python3 -m http.server` works for spot checks, but it has no clean URLs (open `/search.html`).

## Domain

The public origin is set **at build time** with the `SITE_ORIGIN` environment variable
(default `https://ollaya.cobanov.dev`). It is used for canonical and OpenGraph URLs, the install
command, `robots.txt` and `sitemap.xml`; docs use the `{{SITE_ORIGIN}}` placeholder. No host name
is hard-coded anywhere else.

To move to `ollaya.dev`:

1. Add the `ollaya.dev` zone to the same Cloudflare account.
2. In `wrangler.jsonc`, add `{ "pattern": "ollaya.dev", "custom_domain": true }` to `routes`
   (keep the old entry while you redirect, or remove it).
3. `SITE_ORIGIN=https://ollaya.dev npm run deploy`

## Static registry (reserved `/v2/`)

The `/v2/` path prefix is reserved for the future static model registry. `/v2/<ns>/<model>/manifests/<tag>`
files are copied into `dist/` by `convert/package.py`, and blobs are referenced by URL (Hugging Face)
inside the manifests. The site never writes under `dist/v2/` (the build fails if it does), and
`robots.txt` keeps crawlers out of `/v2/`. Unknown `/v2/` paths get the 404 page with status 404.
The build recreates `dist/`, so the order is: `npm run build`, copy the manifests in, then
`npx wrangler deploy`. Plain `npm run deploy` rebuilds and would drop them.

## Deploy

```sh
npx wrangler login     # or: export CLOUDFLARE_API_TOKEN=... ("Edit Cloudflare Workers" token template)
npm run deploy         # optionally: SITE_ORIGIN=https://… npm run deploy
```

This is an assets-only Worker, which is free: static asset requests are not billed and there is no
script to invoke. The custom domain `ollaya.cobanov.dev` requires the `cobanov.dev` zone in the same
Cloudflare account, with no existing DNS record for that host; Wrangler creates the DNS record and
certificate on the first deploy. The `*.workers.dev` URL also stays available (`workers_dev: true`).
