import { textLink } from '../components/ui'
import { catalog, type Model } from '../data/catalog'
import { authorFallback, broughtBy } from '../data/contributors'
import { GITHUB_URL } from '../site'

const cell = 'px-4 py-3 align-top'
const head = 'px-4 py-2.5 font-medium'

for (const name of Object.keys(broughtBy))
  if (!catalog.some((m) => m.name === name)) throw new Error(`contributors: ${name} is not in the catalog`)

/** The catalog's publisher, else a fallback, else the owner of the weights' repository. */
function Author({ model }: { model: Model }) {
  const p = model.publisher ?? authorFallback[model.name] ?? (model.source ? { name: model.source.split('/')[3] ?? model.source, url: model.source } : null)
  return p ? (
    <a href={p.url} class={textLink}>
      {p.name}
    </a>
  ) : (
    <>–</>
  )
}

export function ContributorsPage() {
  return (
    <div class="mx-auto max-w-[52rem] px-4 pt-8 md:px-6 md:pt-12">
      <h1 class="text-[30px] leading-tight font-medium tracking-tight text-fg">Contributors</h1>
      <p class="mt-3 text-lg text-body">
        Ollaya runs other people's models. These are the people who trained them, and the people who brought them here.
      </p>

      <section class="mt-10" aria-labelledby="models-title">
        <h2 id="models-title" class="text-base font-semibold text-fg">
          Models
        </h2>
        <div class="mt-3 overflow-x-auto rounded-lg border border-line">
          <table class="w-full text-left text-sm">
            <thead class="bg-subtle text-[13px] text-muted">
              <tr>
                <th scope="col" class={head}>
                  Model
                </th>
                <th scope="col" class={head}>
                  By
                </th>
                <th scope="col" class={head}>
                  Brought to Ollaya by
                </th>
              </tr>
            </thead>
            <tbody class="divide-y divide-line">
              {catalog.map((m) => (
                <tr>
                  <td class={cell}>
                    <a href={`/library/${m.name}`} class={`font-medium ${textLink}`}>
                      {m.name}
                    </a>
                  </td>
                  <td class={`${cell} text-body`}>
                    <Author model={m} />
                  </td>
                  <td class={`${cell} text-body`}>
                    {(broughtBy[m.name] ?? []).map((b) => (
                      <span class="block">
                        <a href={`https://github.com/${b.github}`} class={textLink}>
                          @{b.github}
                        </a>{' '}
                        {b.what} (
                        {b.refs.map((n, i) => (
                          <>
                            {i ? ', ' : ''}
                            <a href={`${GITHUB_URL}/issues/${n}`} class={textLink}>
                              #{n}
                            </a>
                          </>
                        ))}
                        )
                      </span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section class="mt-10" aria-labelledby="code-title">
        <h2 id="code-title" class="text-base font-semibold text-fg">
          Code
        </h2>
        <p class="mt-3 text-body">
          Platforms, fixes and features come from contributors too: AMD ROCm and Vulkan support, the CUDA pack for
          arm64, Hugging Face mirrors and more. Everyone is on{' '}
          <a href={`${GITHUB_URL}/graphs/contributors`} class={textLink}>
            GitHub's contributors list
          </a>
          .
        </p>
        <p class="mt-3 text-body">
          To add a model or a fix, start with{' '}
          <a href={`${GITHUB_URL}/blob/main/CONTRIBUTING.md`} class={textLink}>
            CONTRIBUTING.md
          </a>
          .
        </p>
      </section>
    </div>
  )
}
