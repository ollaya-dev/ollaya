/**
 * Ollaya's mark: a minimal line-art owl (original artwork). Monochrome — inherits currentColor,
 * so it works on light and dark backgrounds, from 16px favicons to the large closer section.
 * Keep in sync with public/favicon.svg and public/static/logo.svg.
 */
export const OWL_PATHS = {
  head: 'M6 5.5 10.5 9.5Q16 7.5 21.5 9.5L26 5.5V19Q26 27.5 16 27.5T6 19Z',
  beak: 'M14.75 21 16 23l1.25-2',
}

export function LogoMark({ class: cls = 'size-8', label }: { class?: string; label?: string }) {
  return (
    <svg
      class={cls}
      viewBox="0 0 32 32"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : 'true'}
      focusable="false"
    >
      <path d={OWL_PATHS.head} />
      <circle cx="11.5" cy="15.5" r="3.25" />
      <circle cx="20.5" cy="15.5" r="3.25" />
      <circle cx="11.5" cy="15.5" r="1.1" fill="currentColor" stroke="none" />
      <circle cx="20.5" cy="15.5" r="1.1" fill="currentColor" stroke="none" />
      <path d={OWL_PATHS.beak} />
    </svg>
  )
}

/** Mark + lowercase wordmark. */
export function Logo() {
  return (
    <span class="inline-flex items-center gap-2 text-fg">
      <LogoMark class="size-7" />
      <span class="text-lg font-medium tracking-tight">ollaya</span>
    </span>
  )
}
