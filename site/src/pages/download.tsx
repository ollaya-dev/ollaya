import type { Child } from 'hono/jsx'
import { Icon } from '../components/Icon'
import { CodeBlock, PreReleaseNotice, SoonLabel, textLink } from '../components/ui'
import { DOCKER_IMAGE, LOCAL_PORT } from '../site'

type Os = 'linux' | 'macos' | 'docker'

function Panel({ id, selected, children }: { id: Os; selected: Os; children: Child }) {
  return (
    <div role="tabpanel" id={`os-panel-${id}`} aria-labelledby={`os-tab-${id}`} hidden={id !== selected} tabindex={0}>
      {children}
    </div>
  )
}

function Step({ title, soon, children }: { title: string; soon?: boolean; children: Child }) {
  return (
    <div class="mt-8 first:mt-0">
      <div class="flex flex-wrap items-center gap-2">
        <h3 class="text-sm font-semibold text-fg">{title}</h3>
        {soon ? <SoonLabel /> : null}
      </div>
      <div class="mt-3">{children}</div>
    </div>
  )
}

function Requirements({ items }: { items: string[] }) {
  return (
    <Step title="Planned requirements">
      <ul class="list-disc space-y-1.5 pl-5 text-[15px] text-body marker:text-muted">
        {items.map((i) => (
          <li>{i}</li>
        ))}
      </ul>
      <p class="mt-3 text-[13px] text-muted">Final requirements will be published with the first release.</p>
    </Step>
  )
}

/** /download — Linux is selected by default; app.js switches to macOS for Mac visitors. */
export function DownloadPage({ origin }: { origin: string }) {
  const selected: Os = 'linux'
  const install = `curl -fsSL ${origin}/install.sh | sh`
  const tabs: { id: Os; label: string }[] = [
    { id: 'linux', label: 'Linux' },
    { id: 'macos', label: 'macOS' },
    { id: 'docker', label: 'Docker' },
  ]
  const volume = '-v ollaya:/root/.ollaya'
  const port = `-p ${LOCAL_PORT}:${LOCAL_PORT}`

  return (
    <div class="mx-auto max-w-2xl px-4 pt-12 md:px-6 md:pt-20">
      <h1 class="text-center text-4xl font-medium tracking-tight text-fg md:text-5xl">Download Ollaya</h1>
      <p class="mt-4 text-center text-lg text-body">One binary for Linux and macOS, or a Docker image.</p>

      <PreReleaseNotice class="mt-10" />

      <div class="mt-10">
        <div role="tablist" aria-label="Platform" data-os-tabs class="mx-auto flex w-fit gap-1 rounded-full border border-line p-1">
          {tabs.map((t) => {
            const on = t.id === selected
            return (
              <button
                type="button"
                role="tab"
                id={`os-tab-${t.id}`}
                aria-controls={`os-panel-${t.id}`}
                aria-selected={on ? 'true' : 'false'}
                tabindex={on ? 0 : -1}
                class="rounded-full px-5 py-1.5 text-sm font-medium text-muted hover:text-fg aria-selected:bg-btn aria-selected:text-btn-fg"
              >
                {t.label}
              </button>
            )
          })}
        </div>

        <div class="mt-10">
          <Panel id="linux" selected={selected}>
            <Step title="Install with one command" soon>
              <CodeBlock code={install} />
              <p class="mt-3 text-[13px] text-muted">
                Until the first release ships,{' '}
                <a href="/install.sh" class={textLink}>
                  the script
                </a>{' '}
                only prints a notice and exits.
              </p>
            </Step>
            <Requirements
              items={[
                'Runs on the CPU out of the box (ONNX Runtime).',
                'NVIDIA GPUs are accelerated with CUDA.',
                'Roughly 0.65–1.7 GB of disk per model, depending on model and precision.',
              ]}
            />
          </Panel>

          <Panel id="macos" selected={selected}>
            <Step title="Install with one command" soon>
              <CodeBlock code={install} />
              <p class="mt-3 text-[13px] text-muted">
                Until the first release ships,{' '}
                <a href="/install.sh" class={textLink}>
                  the script
                </a>{' '}
                only prints a notice and exits.
              </p>
            </Step>
            <Requirements
              items={[
                'Accelerated with Core ML; runs on the CPU as a fallback.',
                'Roughly 0.65–1.7 GB of disk per model, depending on model and precision.',
              ]}
            />
          </Panel>

          <Panel id="docker" selected={selected}>
            <Step title="CPU only" soon>
              <CodeBlock code={`docker run -d ${port} ${volume} --name ollaya ${DOCKER_IMAGE}`} />
            </Step>
            <Step title="NVIDIA GPU" soon>
              <CodeBlock code={`docker run -d --gpus=all ${port} ${volume} --name ollaya ${DOCKER_IMAGE}`} />
              <p class="mt-3 text-[13px] text-muted">Requires the NVIDIA Container Toolkit on the host.</p>
            </Step>
            <Step title="Run a model" soon>
              <CodeBlock code={`docker exec -it ollaya ollaya run laya --preset triage "I was charged twice this month."`} />
            </Step>
            <p class="mt-6 text-[13px] text-muted">
              The image will be published to <code class="font-mono">{DOCKER_IMAGE}</code> with the first release.
            </p>
          </Panel>
        </div>
      </div>

      <div class="mt-16 grid gap-4 border-t border-line pt-10 sm:grid-cols-2">
        <a href="/docs/quickstart" class="group rounded-lg border border-line p-5 hover:border-line-strong">
          <span class="flex items-center justify-between text-sm font-semibold text-fg">
            Quickstart <Icon name="arrowRight" class="size-4 transition-transform group-hover:translate-x-0.5" />
          </span>
          <span class="mt-1 block text-[15px] text-body">Pull a model, ask typed questions, call the API.</span>
        </a>
        <a href="/search" class="group rounded-lg border border-line p-5 hover:border-line-strong">
          <span class="flex items-center justify-between text-sm font-semibold text-fg">
            Browse models <Icon name="arrowRight" class="size-4 transition-transform group-hover:translate-x-0.5" />
          </span>
          <span class="mt-1 block text-[15px] text-body">Laya in English, 100+ languages, or fine-tuned.</span>
        </a>
      </div>
    </div>
  )
}
