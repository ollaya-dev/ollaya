/**
 * Ollaya's mark: the gaze, two rings with one pupil (an owl reduced to its eyes; the pupil is the
 * option that got chosen). Monochrome: inherits currentColor. Keep in sync with public/favicon.svg,
 * public/static/logo*.svg, scripts/og-lib.mjs and the desktop app (desktop/app-icon.svg,
 * desktop/scripts/tray-icons.swift, GAZE in desktop/ui/src/main.ts).
 */
export const GAZE = {
  left: { cx: 9.5, cy: 16, r: 6.5 },
  right: { cx: 22.5, cy: 16, r: 6.5 },
  pupil: 2.6,
  stroke: 3.2,
}

export function LogoMark({ class: cls = 'size-8', label }: { class?: string; label?: string }) {
  return (
    <svg class={cls} viewBox="0 0 32 32" role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : 'true'} focusable="false">
      <circle cx={GAZE.left.cx} cy={GAZE.left.cy} r={GAZE.left.r} fill="none" stroke="currentColor" stroke-width={GAZE.stroke} />
      <circle cx={GAZE.right.cx} cy={GAZE.right.cy} r={GAZE.right.r} fill="none" stroke="currentColor" stroke-width={GAZE.stroke} />
      <circle cx={GAZE.right.cx} cy={GAZE.right.cy} r={GAZE.pupil} fill="currentColor" />
    </svg>
  )
}

/**
 * Mark + lowercase wordmark. The wordmark is the flex baseline and the mark is centred on its
 * 28px line, so the header can baseline-align the nav links with the wordmark.
 */
export function Logo() {
  return (
    <span class="flex shrink-0 items-baseline gap-2 whitespace-nowrap text-fg">
      <LogoMark class="size-7 self-center" />
      <span class="text-lg leading-7 font-semibold tracking-[-0.035em]">
        ollaya
        {/* vertical-align: middle centres the badge on the wordmark's x-height, whatever the font. */}
        <span
          class="ml-2 inline-block rounded-full border border-line-strong px-[7px] align-middle text-[11px] leading-[17px] font-medium tracking-normal text-muted"
          title="Ollaya is new: expect rough edges, and please report what breaks on GitHub"
        >
          Beta
        </span>
      </span>
    </span>
  )
}
