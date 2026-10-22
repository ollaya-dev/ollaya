import { Icon } from '../components/Icon'
import { btnPrimary, btnSecondary, CodeTabs, OutlinePill, SizeBadge, textLink } from '../components/ui'
import { EXAMPLES, MODEL, type TryExample } from '../data/try'
import { asset } from '../lib/assets'
import { LOCAL_API } from '../site'

// The page is static; src/client/try.js (bundled to /static/try.js by scripts/build.mjs) loads the model on the
// button, renders the example's questions and the answers. The first example is rendered here too, so the page
// has its final shape before the script runs.

const first = EXAMPLES[0]!
const modelName = `${MODEL.name}:${MODEL.tag}`
const hubUrl = `https://huggingface.co/${MODEL.repo}`

const npmCode = `npm install @ollaya/web

import { fromHub, load } from '@ollaya/web'
const model = await load(fromHub('${MODEL.repo}'))
const { answers } = await model.decide({ state, questions })`

const scriptCode = `<script type="importmap">
  { "imports": { "onnxruntime-web": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.bundle.min.mjs" } }
</script>
<script type="module">
  import { fromHub, load } from 'https://cdn.jsdelivr.net/npm/@ollaya/web@0.1.0/ollaya.js'
  const model = await load(fromHub('${MODEL.repo}'))
  console.log(await model.decide({ state, questions }))
</script>`

const nativeCode = `ollaya pull ${modelName}

curl ${LOCAL_API}/api/decide -d '{
  "model": "${modelName}",
  "state": "I was charged twice. Please refund the duplicate.",
  "questions": {
    "refund": {"type": "noul", "instructions": "The customer asks for money back."}
  }
}'`

const TYPE_LABEL = { noul: 'yes / no', choice: 'choice', score: 'score' } as const

/** The questions of an example, as the client also draws them (keep the markup in step with try.js). */
function Questions({ example }: { example: TryExample }) {
  return (
    <ul class="divide-y divide-line" role="list">
      {Object.entries(example.questions).map(([id, q]) => (
        <li class="py-3 first:pt-0 last:pb-0">
          <p class="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span class="font-mono text-[13px] text-fg">{id}</span>
            <OutlinePill>{TYPE_LABEL[q.type]}</OutlinePill>
          </p>
          <p class="mt-1 text-sm text-body">{q.instructions}</p>
          {q.type === 'noul' ? null : (
            <p class="mt-1.5 text-[13px] text-muted">
              {(Array.isArray(q.criteria) ? q.criteria : Object.keys(q.criteria)).join(' · ')}
            </p>
          )}
        </li>
      ))}
    </ul>
  )
}

/** The answers before the first run: each question with an empty bar, the same height as its answer. */
function Pending({ example }: { example: TryExample }) {
  return (
    <ul class="divide-y divide-line" role="list">
      {Object.entries(example.questions).map(([id]) => (
        <li class="py-4 first:pt-0 last:pb-0">
          <p class="flex items-baseline justify-between gap-3">
            <span class="font-mono text-[13px] text-fg">{id}</span>
            <span class="text-sm text-muted tabular-nums">–</span>
          </p>
          <span class="mt-2 block h-1.5 rounded-full bg-fill-strong" aria-hidden="true"></span>
        </li>
      ))}
    </ul>
  )
}

const label = 'text-[13px] font-medium text-muted'
const card = 'rounded-2xl border border-line'

export function TryPage() {
  return (
    <>
      <section class="mx-auto max-w-6xl px-4 pt-10 md:px-6 md:pt-20" aria-labelledby="try-title">
        <h1 id="try-title" class="max-w-3xl text-4xl leading-[1.05] font-medium tracking-tight text-fg md:text-5xl">
          Decisions computed on your device.
        </h1>
        <p class="mt-5 max-w-2xl text-lg text-body">
          {modelName} is Ollaya's own open decision model: {MODEL.size} of ternary weights under Apache-2.0, running in this
          tab with WebAssembly, on a laptop or a phone. Nothing you type is sent to a server.
        </p>
      </section>

      <div
        id="try-app"
        class="mx-auto mt-12 max-w-6xl px-4 md:mt-16 md:px-6"
        data-repo={MODEL.repo}
        data-revision={MODEL.revision}
        data-name={modelName}
        data-runtime-wasm={asset('ollaya-wasm.js')}
        data-ollaya-wasm={asset('ollaya_web_bg.wasm')}
        data-ort-wasm={asset('ort-wasm-simd-threaded.wasm')}
        data-ort-mjs={asset('ort-wasm-simd-threaded.mjs')}
      >
        {/* The model: what it is, its download and the backend it runs on. */}
        <section aria-labelledby="model-title" class={`${card} p-5 sm:p-6`}>
          <div class="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
            <div class="min-w-0">
              <h2 id="model-title" class="flex flex-wrap items-center gap-2">
                <span class="font-mono text-[15px] font-medium text-fg" data-model-name>
                  {modelName}
                </span>
                <SizeBadge>{MODEL.size}</SizeBadge>
                <OutlinePill>Apache-2.0</OutlinePill>
              </h2>
              <p class="mt-1 truncate text-[13px] text-muted">
                Weights from{' '}
                <a href={hubUrl} class={textLink} data-model-repo>
                  huggingface.co/{MODEL.repo}
                </a>
              </p>
            </div>
            <div class="flex flex-wrap items-center gap-x-4 gap-y-3">
              <button type="button" class={`${btnPrimary} disabled:opacity-50`} data-load>
                <Icon name="download" class="size-4" />
                Load the model ({MODEL.size})
              </button>
            </div>
          </div>
          <div class="mt-5">
            <div
              class="h-1.5 overflow-hidden rounded-full bg-fill-strong"
              role="progressbar"
              aria-label="Download"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={0}
              data-progress
            >
              <span class="block h-full w-0 rounded-full bg-us transition-[width] duration-150" data-progress-bar></span>
            </div>
            <p class="mt-2 min-h-5 text-[13px] text-muted tabular-nums" aria-live="polite" data-status>
              Not loaded. Nothing downloads until you press the button.
            </p>
          </div>
        </section>

        <div class="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
          {/* The request: an example, its state (editable) and its questions. */}
          <section aria-labelledby="request-title" class={`${card} p-5 sm:p-6`}>
            <h2 id="request-title" class="sr-only">
              Request
            </h2>
            <p class={label} id="examples-label">
              Examples
            </p>
            <div class="mt-2 flex flex-wrap gap-2" role="group" aria-labelledby="examples-label">
              {EXAMPLES.map((e, i) => (
                <button
                  type="button"
                  data-example={e.key}
                  aria-pressed={i === 0 ? 'true' : 'false'}
                  class="rounded-full border border-line px-3 py-1.5 text-[13px] font-medium text-body hover:border-line-strong hover:text-fg aria-pressed:border-fg aria-pressed:bg-fg aria-pressed:text-canvas"
                >
                  {e.label}
                </button>
              ))}
            </div>

            <label for="try-state" class={`mt-6 block ${label}`}>
              State
            </label>
            <textarea
              id="try-state"
              rows={4}
              spellcheck={false}
              class="mt-2 block w-full resize-y rounded-lg border border-line bg-canvas px-3.5 py-3 text-[15px] leading-relaxed text-fg hover:border-line-strong focus:border-line-strong focus:ring-4 focus:ring-fill-strong focus:outline-none"
              data-state
            >
              {first.state}
            </textarea>

            <p class={`mt-6 ${label}`}>Questions</p>
            <div class="mt-3" data-questions>
              <Questions example={first} />
            </div>

            <details class="group mt-6 border-t border-line pt-4">
              <summary class="inline-flex cursor-pointer items-center gap-1.5 text-sm font-medium text-fg">
                <Icon name="chevronDown" class="size-4 -rotate-90 transition-transform group-open:rotate-0" />
                Advanced: edit the request as JSON
              </summary>
              <label for="try-json" class="sr-only">
                Request body
              </label>
              <textarea
                id="try-json"
                rows={14}
                spellcheck={false}
                class="mt-3 block w-full resize-y rounded-lg border border-line bg-code px-3.5 py-3 font-mono text-[12.5px] leading-relaxed text-fg focus:border-line-strong focus:ring-4 focus:ring-fill-strong focus:outline-none"
                aria-describedby="try-json-error"
                data-json
              >
                {JSON.stringify({ state: first.state, questions: first.questions }, null, 2)}
              </textarea>
              <p id="try-json-error" class="mt-2 min-h-5 text-[13px] text-fail" data-json-error></p>
            </details>
          </section>

          {/* The answers. */}
          <section aria-labelledby="answers-title" class={`${card} flex flex-col p-5 sm:p-6`}>
            <div class="flex items-center justify-between gap-3">
              <h2 id="answers-title" class={label}>
                Answers
              </h2>
              <p class="text-[13px] text-muted tabular-nums" data-latency></p>
            </div>
            <div class="mt-3 flex-1" aria-live="polite" data-answers>
              <Pending example={first} />
            </div>
            <div class="mt-6 flex flex-wrap items-center gap-x-4 gap-y-2">
              <button type="button" class={`${btnPrimary} disabled:opacity-50`} data-decide disabled>
                Decide
              </button>
              <button type="button" class={btnSecondary} data-share>
                <Icon name="copy" class="size-4" />
                <span data-share-label>Copy link</span>
              </button>
              <p class="text-[13px] text-muted" data-decide-hint>
                Load the model first.
              </p>
            </div>
          </section>
        </div>

        <p class="mt-6 flex max-w-3xl gap-2.5 text-[13px] leading-relaxed text-muted">
          <Icon name="info" class="mt-px size-4 shrink-0" />
          <span>
            The text never leaves this page: the tokenizer, the graph and the answers all run in your browser. Only the
            model's files are downloaded, unmodified, from its author's Hugging Face repository.
          </span>
        </p>
        <script type="application/json" id="try-examples" dangerouslySetInnerHTML={{ __html: JSON.stringify(EXAMPLES).replace(/</g, '\\u003c') }}></script>
      </div>

      <section class="mx-auto mt-24 max-w-6xl px-4 md:mt-32 md:px-6" aria-labelledby="use-title">
        <h2 id="use-title" class="text-base font-semibold text-fg">
          Use it in your app
        </h2>
        <p class="mt-2 max-w-2xl text-2xl leading-tight font-medium tracking-tight text-fg md:text-3xl">
          The same model, three lines away.
        </p>
        <p class="mt-4 max-w-2xl text-base text-body md:text-lg">
          In the browser with <code class="font-mono text-[0.9em]">@ollaya/web</code>, or natively with Ollaya on a CPU or
          GPU. Both take the same <code class="font-mono text-[0.9em]">/api/decide</code> request.
        </p>
        <div class="mt-8 max-w-3xl">
          <CodeTabs
            id="use"
            label="Ways to run the model"
            tabs={[
              { key: 'npm', label: 'npm', code: npmCode, lang: 'javascript' },
              { key: 'script', label: 'Script tag', code: scriptCode, lang: 'javascript' },
              { key: 'native', label: 'Ollaya app', code: nativeCode, lang: 'shell' },
            ]}
          />
        </div>
        <p class="mt-6 text-sm">
          <a href="/docs/api" class="inline-flex items-center gap-1.5 font-medium text-fg underline-offset-4 hover:underline">
            The request and answer format <Icon name="arrowRight" class="size-4" />
          </a>
        </p>
      </section>
      <script type="module" src={asset('try.js')}></script>
    </>
  )
}
