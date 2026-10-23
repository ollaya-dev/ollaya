import { textLink } from '../components/ui'
import { blogPosts, type BlogPost } from '../generated/content'
import { SITE_NAME } from '../site'

/** "2026-10-23" → "October 23, 2026". */
const longDate = (date: string) =>
  new Date(`${date}T00:00:00Z`).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' })

export function BlogIndex() {
  return (
    <div class="max-w-[64rem] px-6 pt-10 pb-16 md:px-10 md:pt-14">
      <h1 class="text-4xl font-semibold leading-[1.05] tracking-[-0.03em] text-fg md:text-5xl">Blog</h1>
      <p class="mt-3 text-lg text-body">
        Releases, models and measurements from Ollaya.{' '}
        <a href="/blog/rss.xml" class={textLink}>
          RSS
        </a>
      </p>
      <ul class="mt-10 divide-y divide-line border-y border-line" role="list">
        {blogPosts.map((p) => (
          <li>
            <a href={`/blog/${p.slug}`} class="group block py-6">
              <time datetime={p.date} class="text-[13px] text-muted">
                {longDate(p.date)}
              </time>
              <span class="mt-1 block text-xl font-medium text-fg group-hover:underline underline-offset-4">{p.title}</span>
              <span class="mt-1.5 block text-body">{p.summary}</span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** The post's h2 headings, for the table of contents (the Markdown renderer gives each an id). */
const headings = (html: string) => [...html.matchAll(/<h2 id="([^"]+)"[^>]*>(?:<a[^>]*>)?([^<]+)/g)].map((m) => ({ id: m[1]!, text: m[2]! }))

export function BlogPostView({ post, html }: { post: BlogPost; html: string }) {
  const toc = headings(html)
  return (
    <div class="px-6 pt-10 pb-16 md:px-10 md:pt-14 lg:grid lg:grid-cols-[minmax(0,1fr)_16rem] lg:gap-14">
      <div class="min-w-0 max-w-[46rem]">
        <a href="/blog" class="text-[13px] text-muted underline-offset-4 hover:text-fg hover:underline">
          Blog
        </a>
        <h1 class="mt-3 text-4xl font-semibold leading-[1.05] tracking-[-0.03em] text-fg md:text-5xl">{post.title}</h1>
        <time datetime={post.date} class="mt-3 block text-[13px] text-muted">
          {longDate(post.date)}
        </time>
        <article class="prose mt-8" dangerouslySetInnerHTML={{ __html: html }}></article>
      </div>
      {toc.length ? (
        <nav aria-label="On this page" class="hidden lg:block">
          <ul class="sticky top-24 space-y-2.5 border-l border-line text-sm" data-scrollspy>
            {toc.map((h) => (
              <li>
                <a href={`#${h.id}`} class="-ml-px block border-l-2 border-transparent pl-4 text-muted hover:text-fg aria-[current=true]:border-fg aria-[current=true]:text-fg">
                  {h.text}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      ) : null}
    </div>
  )
}

const xml = (s: string) => s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;')

/** RSS 2.0 feed of every post, newest first, with the summary as each item's description. */
export function rssXml(origin: string): string {
  const items = blogPosts
    .map((p) => {
      const url = `${origin}/blog/${p.slug}`
      return [
        '    <item>',
        `      <title>${xml(p.title)}</title>`,
        `      <link>${url}</link>`,
        `      <guid isPermaLink="true">${url}</guid>`,
        `      <pubDate>${new Date(`${p.date}T00:00:00Z`).toUTCString()}</pubDate>`,
        `      <description>${xml(p.summary)}</description>`,
        '    </item>',
      ].join('\n')
    })
    .join('\n')
  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>${xml(SITE_NAME)} blog</title>
    <link>${origin}/blog</link>
    <description>Releases, models and measurements from Ollaya.</description>
    <language>en</language>
    <atom:link href="${origin}/blog/rss.xml" rel="self" type="application/rss+xml" />
${items}
  </channel>
</rss>
`
}
