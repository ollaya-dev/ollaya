import type { Child } from 'hono/jsx'
import { GITHUB_URL } from '../site'
import { Icon } from './Icon'

export const btnPrimary =
  'inline-flex items-center justify-center gap-2 rounded-full bg-btn px-5 py-2.5 text-sm font-medium text-btn-fg hover:bg-btn-hover'
export const btnSecondary =
  'inline-flex items-center justify-center gap-2 rounded-full border border-line px-5 py-2.5 text-sm font-medium text-fg hover:bg-fill'
export const textLink = 'text-fg underline decoration-line-strong underline-offset-4 hover:decoration-current'
export const quietLink = 'underline-offset-4 hover:text-fg hover:underline'

const badgeBase = 'inline-flex items-center rounded-md px-2 py-[2px] text-[13px] font-medium'

/** Capability badge (indigo). */
export function CapBadge({ children }: { children: Child }) {
  return <span class={`${badgeBase} bg-cap-bg text-cap-fg`}>{children}</span>
}

/** Parameter-size badge (blue). */
export function SizeBadge({ children }: { children: Child }) {
  return <span class={`${badgeBase} bg-size-bg text-size-fg`}>{children}</span>
}

/** Neutral outlined pill, e.g. "latest". */
export function OutlinePill({ children }: { children: Child }) {
  return (
    <span class="inline-flex items-center rounded-full border border-line-strong px-2 py-[1px] text-[11px] font-medium text-muted">
      {children}
    </span>
  )
}

/**
 * Copy-to-clipboard button. public/static/app.js copies the visible <pre> inside the closest
 * [data-copy-scope] and swaps the icon to a check mark for 2 seconds.
 */
export function CopyButton({ class: cls = 'absolute top-2 right-2' }: { class?: string }) {
  return (
    <button
      type="button"
      data-copy
      class={`group inline-flex size-8 shrink-0 items-center justify-center rounded-md text-muted hover:bg-fill hover:text-fg ${cls}`}
      aria-label="Copy code"
    >
      <Icon name="copy" class="size-4 group-data-[copied]:hidden" />
      <Icon name="check" class="hidden size-4 group-data-[copied]:block" />
      <span class="sr-only" data-copy-status></span>
    </button>
  )
}

/** A single code block with a copy button. */
export function CodeBlock({ code, class: cls = '' }: { code: string; class?: string }) {
  return (
    <div class={`relative ${cls}`} data-copy-scope>
      <pre class="overflow-x-auto rounded-lg bg-code py-4 pr-12 pl-4 font-mono text-[13px] leading-relaxed text-fg">
        <code>{code}</code>
      </pre>
      <CopyButton />
    </div>
  )
}

export interface CodeTab {
  key: string
  label: string
  code: string
}

/** Tabbed code box (CLI | cURL | Python | JavaScript, Linux | macOS | Docker, …). */
export function CodeTabs({ id, label, tabs, selected }: { id: string; label: string; tabs: CodeTab[]; selected?: string }) {
  const active = selected ?? tabs[0]?.key
  return (
    <div class="overflow-hidden rounded-lg border border-line" data-copy-scope>
      <div class="flex items-center justify-between gap-2 border-b border-line pr-1.5 pl-2">
        <div role="tablist" aria-label={label} class="-mb-px flex overflow-x-auto">
          {tabs.map((t) => {
            const on = t.key === active
            return (
              <button
                type="button"
                role="tab"
                id={`${id}-tab-${t.key}`}
                aria-controls={`${id}-panel-${t.key}`}
                aria-selected={on ? 'true' : 'false'}
                tabindex={on ? 0 : -1}
                class="border-b-2 border-transparent px-3 py-2.5 text-[13px] font-medium whitespace-nowrap text-muted hover:text-fg aria-selected:border-fg aria-selected:text-fg"
              >
                {t.label}
              </button>
            )
          })}
        </div>
        <CopyButton class="" />
      </div>
      {tabs.map((t) => (
        <div
          role="tabpanel"
          id={`${id}-panel-${t.key}`}
          aria-labelledby={`${id}-tab-${t.key}`}
          hidden={t.key !== active}
          tabindex={0}
          class="bg-code focus-visible:outline-offset-[-2px]"
        >
          <pre class="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed text-fg">
            <code>{t.code}</code>
          </pre>
        </div>
      ))}
    </div>
  )
}

/** Honest pre-release notice used on download, model and docs pages. */
export function PreReleaseNotice({ children, class: cls = '' }: { children?: Child; class?: string }) {
  return (
    <div class={`flex gap-3 rounded-lg border border-line p-4 text-sm ${cls}`} role="note">
      <Icon name="info" class="mt-0.5 size-5 shrink-0 text-muted" />
      <p class="text-body">
        {children ?? (
          <>
            <strong class="font-medium text-fg">Coming soon — first release in progress.</strong> Nothing is
            downloadable yet.{' '}
          </>
        )}{' '}
        <a href={GITHUB_URL} class={textLink}>
          Star / watch on GitHub
        </a>{' '}
        to hear about the first release.
      </p>
    </div>
  )
}

/** Small "Coming soon" label for commands that don't work yet. */
export function SoonLabel() {
  return (
    <span class="inline-flex items-center rounded-md bg-fill px-2 py-[2px] text-[13px] font-medium text-muted">
      Coming soon
    </span>
  )
}
