import type { Child } from 'hono/jsx'
import { Icon } from '../components/Icon'
import { btnPrimary, CodeBlock, textLink } from '../components/ui'
import { DOCKER_IMAGE, LATEST_DOWNLOAD, LOCAL_PORT, RELEASES_URL } from '../site'

type Os = 'macos' | 'windows' | 'linux' | 'docker'

function Panel({ id, selected, children }: { id: Os; selected: Os; children: Child }) {
  return (
    <div role="tabpanel" id={`os-panel-${id}`} aria-labelledby={`os-tab-${id}`} hidden={id !== selected} tabindex={0}>
      {children}
    </div>
  )
}

function Step({ title, children }: { title: string; children: Child }) {
  return (
    <div class="mt-8 first:mt-0">
      <h3 class="text-sm font-semibold text-fg">{title}</h3>
      <div class="mt-3">{children}</div>
    </div>
  )
}

function Requirements({ items }: { items: Child[] }) {
  return (
    <Step title="Requirements">
      <ul class="list-disc space-y-1.5 pl-5 text-[15px] text-body marker:text-muted">
        {items.map((i) => (
          <li>{i}</li>
        ))}
      </ul>
    </Step>
  )
}

function Note({ children }: { children: Child }) {
  return <p class="mt-3 text-[13px] text-muted">{children}</p>
}

/** The desktop app: a download button and what the app does. */
function DesktopApp({ file, label, extra }: { file: string; label: string; extra?: Child }) {
  return (
    <Step title="Desktop app">
      <div class="flex flex-wrap items-center gap-x-4 gap-y-2">
        <a href={`${LATEST_DOWNLOAD}/${file}`} class={btnPrimary}>
          <Icon name="download" class="size-4" />
          {label}
        </a>
        {extra}
      </div>
      <Note>
        Start and stop the server, download models and try them, in one window. Your code talks to the same local API;
        for the <code class="font-mono">ollaya</code> command, install the command line too.
      </Note>
    </Step>
  )
}

/** /download — Linux is selected by default; app.js switches to macOS for Mac visitors. */
export function DownloadPage({ origin }: { origin: string }) {
  const selected: Os = 'linux'
  const install = `curl -fsSL ${origin}/install.sh | sh`
  const installWindows = `irm ${origin}/install.ps1 | iex`
  const tabs: { id: Os; label: string }[] = [
    { id: 'macos', label: 'macOS' },
    { id: 'windows', label: 'Windows' },
    { id: 'linux', label: 'Linux' },
    { id: 'docker', label: 'Docker' },
  ]
  const volume = '-v ollaya:/home/ollaya/.ollaya'
  const port = `-p ${LOCAL_PORT}:${LOCAL_PORT}`
  const script = (
    <a href="/install.sh" class={textLink}>
      The script
    </a>
  )

  return (
    <div class="mx-auto max-w-2xl px-4 pt-12 md:px-6 md:pt-20">
      <h1 class="text-center text-4xl font-medium tracking-tight text-fg md:text-5xl">Download Ollaya</h1>
      <p class="mt-4 text-center text-lg text-body">A desktop app and a command line for macOS, Windows and Linux, or a Docker image.</p>

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
            <Step title="Command line">
              <CodeBlock code={install} />
              <Note>
                {script} detects your CPU and NVIDIA GPU, downloads the release from{' '}
                <a href={RELEASES_URL} class={textLink}>
                  GitHub
                </a>
                , checks its sha256 and, where systemd runs, sets up the <code class="font-mono">ollaya</code>{' '}
                service. With a GPU it also fetches the CUDA libraries (about 1 GB). It never installs drivers.
              </Note>
            </Step>
            <Step title="Run a model">
              <CodeBlock code="ollaya run laya" />
            </Step>
            <Requirements
              items={[
                'x86-64 or ARM64 with glibc 2.38 or newer: Ubuntu 24.04, Debian 13, Fedora 39, RHEL 10 or newer.',
                'Runs on the CPU. An NVIDIA GPU is optional: driver R580 or newer (CUDA 13), on x86-64.',
              ]}
            />
            <DesktopApp
              file="Ollaya-linux-x86_64.AppImage"
              label="Download the AppImage"
              extra={
                <a href={`${LATEST_DOWNLOAD}/Ollaya-linux-amd64.deb`} class={`text-sm ${textLink}`}>
                  or the .deb
                </a>
              }
            />
          </Panel>

          <Panel id="windows" selected={selected}>
            <DesktopApp file="Ollaya-windows-x64-setup.exe" label="Download for Windows" />
            <Step title="Command line">
              <CodeBlock code={installWindows} lang="powershell" />
              <Note>
                In PowerShell. It installs <code class="font-mono">ollaya</code> for your user, with no administrator
                rights, and puts it on your <code class="font-mono">PATH</code>. The archive is checked against the
                release's sha256.
              </Note>
            </Step>
            <Step title="Run a model">
              <CodeBlock code="ollaya run laya" />
            </Step>
            <Requirements
              items={[
                'Windows 10 or 11 on a 64-bit x86 PC. Models run on the CPU.',
                <>
                  For an NVIDIA GPU, use WSL 2 with the Linux installer. The server in WSL answers Windows programs at{' '}
                  <code class="font-mono">localhost:{LOCAL_PORT}</code>.
                </>,
              ]}
            />
          </Panel>

          <Panel id="macos" selected={selected}>
            <DesktopApp file="Ollaya-macos-arm64.dmg" label="Download for macOS" />
            <Step title="Command line">
              <CodeBlock code={install} />
              <Note>
                {script} downloads the release from{' '}
                <a href={RELEASES_URL} class={textLink}>
                  GitHub
                </a>{' '}
                and checks its sha256. Start the server with <code class="font-mono">ollaya serve</code>, or let{' '}
                <code class="font-mono">ollaya run</code> start it for you.
              </Note>
            </Step>
            <Step title="Run a model">
              <CodeBlock code="ollaya run laya" />
            </Step>
            <Requirements items={['A Mac with Apple silicon (arm64).']} />
          </Panel>

          <Panel id="docker" selected={selected}>
            <Step title="CPU">
              <CodeBlock code={`docker run -d --name ollaya ${port} ${volume} ${DOCKER_IMAGE}`} />
            </Step>
            <Step title="NVIDIA GPU">
              <CodeBlock code={`docker run -d --name ollaya --gpus=all ${port} ${volume} ${DOCKER_IMAGE}:cuda`} />
              <Note>Needs the NVIDIA Container Toolkit and a host driver with CUDA 13 support (R580 or newer).</Note>
            </Step>
            <Step title="Run a model">
              <CodeBlock code="docker exec -it ollaya ollaya run laya" />
            </Step>
            <Note>
              The CPU image is built for linux/amd64 and linux/arm64, the <code class="font-mono">:cuda</code> image
              for linux/amd64. Models are kept in the <code class="font-mono">ollaya</code> volume.
            </Note>
          </Panel>
        </div>
      </div>

      <p class="mt-10 text-center text-[13px] text-muted">
        Prefer a tarball? Every release on{' '}
        <a href={RELEASES_URL} class={textLink}>
          GitHub Releases
        </a>{' '}
        has the archives and a <code class="font-mono">sha256sum.txt</code>.
      </p>

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
