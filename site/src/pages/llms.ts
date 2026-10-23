/**
 * /llms.txt and /llms-full.txt (https://llmstxt.org): what Ollaya is and where its docs are, for AI
 * assistants and answer engines. Built from the same docs and catalog as the pages.
 */
import { withResults } from '../data/numbers'
import type { Model } from '../data/catalog'
import type { DocPage } from '../generated/content'
import { GITHUB_URL, SITE_DESCRIPTION, SITE_NAME } from '../site'
import { withOrigin } from './docs'

const intro = (origin: string) =>
  `A decision model reads a state (a message, an email, a ticket, any JSON) plus typed questions (choice, score, yes/no) and returns calibrated probabilities in a single forward pass, in milliseconds. It never generates text. Ollaya pulls these models by name from their authors' Hugging Face repositories, serves them from a local daemon on port 11435, and speaks TypeSafe's \`/v1/systemone\` wire format, so the TypeSafe SDK and other Jev clients work by changing one environment variable. Install: \`curl -fsSL ${origin}/install.sh | sh\`.`

export function llmsTxt(origin: string, bench: string, docs: DocPage[], models: Model[]): string {
  const link = (title: string, url: string, about: string) => `- [${title}](${url}): ${about}`
  return [
    `# ${SITE_NAME}`,
    `> ${SITE_DESCRIPTION}`,
    intro(origin),
    ['## Docs', ...docs.map((d) => link(d.title, `${origin}/docs/${d.slug}`, d.description))].join('\n'),
    ['## Models', ...models.map((m) => link(m.name, `${origin}/library/${m.name}`, m.description))].join('\n'),
    [
      '## Optional',
      link('Results', `${origin}/results`, 'accuracy, calibration, speed and parity of every model, with Ollama on the same GPU'),
      link('Ollaya Bench', bench, 'the public leaderboard of open decision models'),
      link('Source code', GITHUB_URL, 'Apache-2.0'),
      link('Full docs', `${origin}/llms-full.txt`, 'every docs page above in one Markdown file'),
    ].join('\n'),
  ].join('\n\n') + '\n'
}

/** Every docs page as Markdown, numbers filled in and site links made absolute. */
export function llmsFullTxt(origin: string, docs: DocPage[]): string {
  const pages = docs.map((d) => {
    const md = withResults(withOrigin(d.md, origin)).replace(/\]\(\//g, `](${origin}/`)
    return `<!-- ${origin}/docs/${d.slug} -->\n\n${md.trim()}\n`
  })
  return `# ${SITE_NAME}\n\n> ${SITE_DESCRIPTION}\n\n${intro(origin)}\n\n${pages.join('\n')}`
}
