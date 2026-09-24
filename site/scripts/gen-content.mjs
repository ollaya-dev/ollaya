// Build-time Markdown rendering (no Markdown parser ships to the browser).
//
//   docs/*.md              -> src/generated/content.ts  (docs pages, with front matter)
//   content/library/*.md   -> src/generated/content.ts  (model readmes)
//
// Markdown may contain the placeholder {{SITE_ORIGIN}}; the static build substitutes the
// configured origin (SITE_ORIGIN env var) so no host name is ever baked into content.

import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Marked } from 'marked'
import { highlight } from '../src/lib/highlight.ts'

const root = fileURLToPath(new URL('..', import.meta.url))
const outDir = join(root, 'src', 'generated')

const escapeHtml = (s) =>
  String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')

const stripTags = (s) => s.replace(/<[^>]*>/g, '')

const slugify = (s) =>
  stripTags(s)
    .toLowerCase()
    .replace(/&[a-z]+;/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/^\d+-(?=[a-z])/, '')

function parseFrontMatter(src, file) {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(src)
  if (!m) return { data: {}, body: src }
  const data = {}
  for (const line of m[1].split(/\r?\n/)) {
    if (!line.trim()) continue
    const i = line.indexOf(':')
    if (i < 0) throw new Error(`${file}: bad front matter line: ${line}`)
    data[line.slice(0, i).trim()] = line.slice(i + 1).trim().replace(/^"(.*)"$/, '$1')
  }
  return { data, body: src.slice(m[0].length) }
}

// Copy button markup for Markdown code blocks. Keep in sync with CopyButton in src/components/ui.tsx.
const copyButton = `<button type="button" data-copy class="group absolute top-2 right-2 inline-flex size-8 items-center justify-center rounded-md text-muted hover:bg-fill hover:text-fg" aria-label="Copy code"><svg class="size-4 group-data-[copied]:hidden" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M16.5 8.25V6a2.25 2.25 0 0 0-2.25-2.25H6A2.25 2.25 0 0 0 3.75 6v8.25A2.25 2.25 0 0 0 6 16.5h2.25m8.25-8.25H18a2.25 2.25 0 0 1 2.25 2.25V18A2.25 2.25 0 0 1 18 20.25h-7.5A2.25 2.25 0 0 1 8.25 18v-1.5m8.25-8.25h-6a2.25 2.25 0 0 0-2.25 2.25v6"/></svg><svg class="hidden size-4 group-data-[copied]:block" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5"/></svg><span class="sr-only" data-copy-status></span></button>`

function createRenderer({ demote = 0 } = {}) {
  let seen = new Map()
  const marked = new Marked({
    gfm: true,
    renderer: {
      heading({ tokens, depth }) {
        const inner = this.parser.parseInline(tokens)
        const level = Math.min(6, depth + demote)
        let id = slugify(inner) || 'section'
        const n = seen.get(id) ?? 0
        seen.set(id, n + 1)
        if (n) id = `${id}-${n}`
        if (level === 1) return `<h1 id="${id}">${inner}</h1>\n`
        return `<h${level} id="${id}"><a href="#${id}">${inner}</a></h${level}>\n`
      },
      code({ text, lang }) {
        const language = (lang || '').trim().split(/\s+/)[0]
        const cls = language ? ` class="language-${escapeHtml(language)}"` : ''
        return `<div class="codeblock relative" data-copy-scope><pre><code${cls}>${highlight(text, language)}</code></pre>${copyButton}</div>\n`
      },
      table(token) {
        // Default rendering, wrapped for horizontal scrolling on narrow screens.
        const header = token.header
          .map((cell) => `<th${cell.align ? ` style="text-align:${cell.align}"` : ''}>${this.parser.parseInline(cell.tokens)}</th>`)
          .join('')
        const rows = token.rows
          .map(
            (row) =>
              `<tr>${row
                .map((cell) => `<td${cell.align ? ` style="text-align:${cell.align}"` : ''}>${this.parser.parseInline(cell.tokens)}</td>`)
                .join('')}</tr>`,
          )
          .join('\n')
        return `<div class="table-wrap"><table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table></div>\n`
      },
    },
  })
  return {
    render(md) {
      seen = new Map()
      return marked.parse(md, { async: false })
    },
  }
}

async function listFiles(dir, filter) {
  let entries
  try {
    entries = await readdir(dir, { withFileTypes: true })
  } catch {
    return []
  }
  const out = []
  for (const e of entries) {
    const p = join(dir, e.name)
    if (e.isDirectory()) out.push(...(await listFiles(p, filter)))
    else if (!filter || filter(e.name)) out.push(p)
  }
  return out.sort()
}

async function buildDocs() {
  const renderer = createRenderer()
  const files = await listFiles(join(root, 'docs'), (n) => n.endsWith('.md'))
  const pages = []
  for (const file of files) {
    const slug = file.slice(file.lastIndexOf('/') + 1, -3)
    const { data, body } = parseFrontMatter(await readFile(file, 'utf8'), file)
    for (const key of ['title', 'description', 'order']) {
      if (!data[key]) throw new Error(`${relative(root, file)}: missing front matter "${key}"`)
    }
    pages.push({
      slug,
      title: data.title,
      nav: data.nav || data.title,
      description: data.description,
      order: Number(data.order),
      html: renderer.render(body),
    })
  }
  return pages.sort((a, b) => a.order - b.order)
}

async function buildReadmes() {
  const renderer = createRenderer({ demote: 1 })
  const files = await listFiles(join(root, 'content', 'library'), (n) => n.endsWith('.md'))
  const out = {}
  for (const file of files) {
    const name = file.slice(file.lastIndexOf('/') + 1, -3)
    out[name] = renderer.render(await readFile(file, 'utf8'))
  }
  return out
}

const header = '// AUTO-GENERATED by scripts/gen.mjs — do not edit. Run `npm run gen`.\n\n'

const [docs, readmes] = await Promise.all([buildDocs(), buildReadmes()])
await mkdir(outDir, { recursive: true })

await writeFile(
  join(outDir, 'content.ts'),
  header +
    `export interface DocPage {\n  slug: string\n  title: string\n  nav: string\n  description: string\n  order: number\n  html: string\n}\n\n` +
    `export const docPages: DocPage[] = ${JSON.stringify(docs, null, 2)}\n\n` +
    `export const readmes: Record<string, string> = ${JSON.stringify(readmes, null, 2)}\n`,
)

console.log(`content: ${docs.length} docs (${docs.map((d) => d.slug).join(', ')}), ${Object.keys(readmes).length} readme(s)`)
