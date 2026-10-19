/**
 * Charts for the results page, drawn at build time as inline SVG (no client script, no chart
 * library). Colour comes from the data tokens in app.css (Ollaya against another server, GPUs
 * against CPUs, pass against fail), every chart has a legend for its colours and marks, and the
 * numbers are on the page as text too.
 */
import type { Child } from 'hono/jsx'

const fmt = (v: number, digits = 3) => v.toFixed(digits)

/** A data colour by token name (app.css): Ollaya or another server, a GPU or a CPU, Vulkan, a parity check. */
export type Tone = 'us' | 'them' | 'gpu' | 'cpu' | 'vulkan' | 'pass' | 'fail'
export const toneVar = (t: Tone) => `var(--color-${t})`

/** A number for an axis tick: 5, 20, 100, 1k, 10k. */
export function tickLabel(ms: number): string {
  if (ms >= 1000) return `${ms / 1000}k`.replace('.0k', 'k')
  return `${ms}`
}

type Scale = (v: number) => number

function logScale(min: number, max: number, from: number, to: number): Scale {
  const a = Math.log(min)
  const b = Math.log(max)
  return (v) => from + ((Math.log(Math.max(min, Math.min(max, v))) - a) / (b - a)) * (to - from)
}

function linScale(min: number, max: number, from: number, to: number): Scale {
  return (v) => from + ((v - min) / (max - min)) * (to - from)
}

/**
 * The classes and colour tokens the SVG charts use, as plain CSS: embedded in a chart published
 * as its own .svg file (the README shows them), where the site's stylesheet does not reach.
 * Keep the colours in step with app.css.
 */
const STANDALONE_CSS = `
text{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
svg{--color-us:#0969da;--color-them:#bc4c00;--color-gpu:#0969da;--color-cpu:#bc4c00;--color-vulkan:#8250df;--color-pass:#1a7f37;--color-fail:#cf222e}
.fill-fg{fill:#171717}.fill-body{fill:#262626}.fill-muted{fill:#737373}.fill-canvas{fill:#fff}
.stroke-line{stroke:#e5e5e5}.stroke-line-strong{stroke:#d4d4d4}.stroke-muted{stroke:#737373}.stroke-canvas{stroke:#fff}
.bg{fill:#fff}
@media (prefers-color-scheme:dark){
svg{--color-us:#4493f8;--color-them:#f0883e;--color-gpu:#4493f8;--color-cpu:#f0883e;--color-vulkan:#ab7df8;--color-pass:#3fb950;--color-fail:#f85149}
.fill-fg{fill:#fafafa}.fill-body{fill:#d4d4d4}.fill-muted{fill:#a3a3a3}.fill-canvas{fill:#0d1117}
.stroke-line{stroke:#262626}.stroke-line-strong{stroke:#404040}.stroke-muted{stroke:#a3a3a3}.stroke-canvas{stroke:#0d1117}
.bg{fill:#0d1117}}
`

/** Attributes and the leading children that make a chart a standalone .svg file. */
function standaloneParts(W: number, H: number, on: boolean | undefined) {
  if (!on) return { attrs: {}, head: null }
  return {
    attrs: { xmlns: 'http://www.w3.org/2000/svg', width: `${W}`, height: `${H}` },
    head: (
      <>
        <style>{STANDALONE_CSS}</style>
        <rect width={W} height={H} class="bg" />
      </>
    ),
  }
}

// --------------------------------------------------------------------------------------------
// Marks and legends

/** How a series is drawn: a filled dot, or a ring (an open circle). */
export type Mark = 'dot' | 'ring'

export interface SeriesKey {
  label: string
  tone: Tone
  mark?: Mark
}

function MarkSvg({ cx, cy, r, tone, mark }: { cx: number; cy: number; r: number; tone: Tone; mark?: Mark }) {
  return mark === 'ring' ? (
    <circle cx={cx} cy={cy} r={r - 1} class="fill-canvas" style={`stroke:${toneVar(tone)}`} stroke-width="2.5" />
  ) : (
    <circle cx={cx} cy={cy} r={r} style={`fill:${toneVar(tone)}`} class="stroke-canvas" stroke-width="1.5" />
  )
}

/** The legend above a chart: one entry per colour and mark. */
export function Legend({ items }: { items: SeriesKey[] }) {
  return (
    <ul class="flex flex-wrap gap-x-5 gap-y-2 text-[13px] text-body" role="list">
      {items.map((s) => (
        <li class="inline-flex items-center gap-2">
          <svg viewBox="0 0 12 12" class="size-3" aria-hidden="true">
            <MarkSvg cx={6} cy={6} r={5} tone={s.tone} mark={s.mark} />
          </svg>
          {s.label}
        </li>
      ))}
    </ul>
  )
}

/** The same legend inside an SVG (for the standalone files); returns its height. */
function SvgLegend({ items, x, y }: { items: SeriesKey[]; x: number; y: number }) {
  let at = x
  return (
    <g>
      {items.map((s) => {
        const left = at
        at += 30 + s.label.length * 6.6
        return (
          <g>
            <MarkSvg cx={left + 6} cy={y} r={5} tone={s.tone} mark={s.mark} />
            <text x={left + 16} y={y + 4} font-size="12" class="fill-body">
              {s.label}
            </text>
          </g>
        )
      })}
    </g>
  )
}

// --------------------------------------------------------------------------------------------
// Scatter: accuracy against latency

export interface ScatterPoint {
  label: string
  x: number
  y: number
  tone: Tone
  mark?: Mark
}

interface Box {
  x: number
  y: number
  w: number
  h: number
}
type Seg = [number, number, number, number]

const overlaps = (a: Box, b: Box) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h

function cross(a: Seg, b: Seg): boolean {
  const o = (px: number, py: number, qx: number, qy: number, rx: number, ry: number) => Math.sign((qx - px) * (ry - py) - (qy - py) * (rx - px))
  const [ax, ay, bx, by] = a
  const [cx, cy, dx, dy] = b
  return o(ax, ay, bx, by, cx, cy) !== o(ax, ay, bx, by, dx, dy) && o(cx, cy, dx, dy, ax, ay) !== o(cx, cy, dx, dy, bx, by)
}

function segHitsBox([x1, y1, x2, y2]: Seg, b: Box): boolean {
  for (let i = 1; i < 12; i++) {
    const t = i / 12
    const x = x1 + (x2 - x1) * t
    const y = y1 + (y2 - y1) * t
    if (x > b.x && x < b.x + b.w && y > b.y && y < b.y + b.h) return true
  }
  return false
}

/**
 * Points with their labels. Each label goes in the nearest free spot around its point (right,
 * left, above, below, then the diagonals, further out each round); a label away from its point
 * gets a short leader line, and no leader crosses another leader, a label or a point. Crowded
 * points are labelled first, while the space around them is still free. A label with no free
 * spot is left out rather than drawn over another; every point names itself on hover.
 */
export function Scatter({
  points,
  xTicks,
  yTicks,
  xTitle,
  yTitle,
  label,
  legend,
  standalone,
  compact,
}: {
  points: ScatterPoint[]
  xTicks: number[]
  yTicks: number[]
  xTitle: string
  yTitle: string
  label: string
  legend?: SeriesKey[]
  /** Render as its own .svg file, with its colours and the legend inside. */
  standalone?: boolean
  /** The phone layout: a narrower canvas, so the text keeps its size on a small screen. */
  compact?: boolean
}) {
  const W = compact ? 400 : 760
  const top = standalone && legend ? 40 : 16
  const H = (compact ? 520 : 540) + top - 16
  const m = { l: compact ? 40 : 52, r: compact ? 12 : 16, t: top, b: compact ? 44 : 52 }
  const fs = compact ? 10.5 : 11.5
  const x = logScale(xTicks[0]!, xTicks.at(-1)!, m.l, W - m.r)
  const y = linScale(yTicks[0]!, yTicks.at(-1)!, H - m.b, m.t)
  const charW = fs * 0.61 // the advance of a monospace glyph, with a little to spare
  const lh = compact ? 13 : 14
  const pts = points.map((p) => ({ p, px: x(p.x), py: y(p.y) }))
  const pointBoxes: Box[] = pts.map(({ px, py }) => ({ x: px - 7, y: py - 7, w: 14, h: 14 }))
  const boxes: Box[] = [...pointBoxes]
  const leaders: Seg[] = []
  const crowd = (a: (typeof pts)[number]) => pts.filter((b) => Math.hypot(a.px - b.px, a.py - b.py) < 60).length
  const dirs: [number, number][] = [
    [1, 0],
    [-1, 0],
    [0, -1],
    [0, 1],
    [1, -1],
    [1, 1],
    [-1, -1],
    [-1, 1],
  ]
  const labels = [...pts]
    .sort((a, b) => crowd(b) - crowd(a) || a.py - b.py)
    .map(({ p, px, py }) => {
      const w = p.label.length * charW
      const self = pointBoxes[pts.findIndex((q) => q.p === p)]
      for (const d of [9, 14, 20, 26, 32, 40, 48, 58]) {
        for (const [dx, dy] of dirs) {
          const bx = dx > 0 ? px + d : dx < 0 ? px - d - w : px - w / 2
          const by = dy > 0 ? py + d - 4 : dy < 0 ? py - d - lh + 4 : py - lh / 2
          const box = { x: bx, y: by, w, h: lh }
          if (box.x < m.l + 2 || box.x + w > W - 2 || box.y < m.t - 12 || box.y + lh > H - m.b) continue
          // A few pixels of air, so two labels never touch end to end, and no label sits on another's leader.
          const air = { x: bx - 4, y: by - 1, w: w + 8, h: lh + 2 }
          if (boxes.some((b) => overlaps(b === self ? box : air, b)) || leaders.some((s) => segHitsBox(s, air))) continue
          // A label without a leader is closer to its own point than to any other, so a reader pairs them right.
          const near = (q: { px: number; py: number }) => Math.hypot(q.px - (bx + w / 2), q.py - (by + lh / 2))
          if (d === 9 && pts.some((q) => q.p !== p && near(q) < near({ px, py }))) continue
          const ex = Math.max(bx, Math.min(px, bx + w))
          const ey = Math.max(by, Math.min(py, by + lh))
          const lead = d > 9
          if (lead) {
            const seg: Seg = [px, py, ex, ey]
            if (leaders.some((s) => cross(s, seg))) continue
            if (boxes.some((b) => b !== self && segHitsBox(seg, b))) continue
            leaders.push(seg)
          }
          boxes.push(box)
          return { p, x: bx, y: by + 11, lead, ex, ey, px, py }
        }
      }
      return null
    })
    .filter((l) => l !== null)
  const sa = standaloneParts(W, H, standalone)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} class="h-auto w-full" role="img" aria-label={label} {...sa.attrs}>
      <title>{label}</title>
      {sa.head}
      {standalone && legend ? <SvgLegend items={legend} x={m.l} y={16} /> : null}
      {yTicks.map((t) => (
        <g>
          <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} class="stroke-line" stroke-width="1" />
          <text x={m.l - 6} y={y(t) + 4} text-anchor="end" class="fill-muted" font-size={compact ? 10 : 11}>
            {compact ? t.toFixed(1) : t.toFixed(2)}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <g>
          <line x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} class="stroke-line" stroke-width="1" />
          <text x={x(t)} y={H - m.b + 16} text-anchor="middle" class="fill-muted" font-size={compact ? 10 : 11}>
            {tickLabel(t)}
          </text>
        </g>
      ))}
      <text x={(m.l + W - m.r) / 2} y={H - 10} text-anchor="middle" class="fill-muted" font-size={compact ? 10.5 : 12}>
        {xTitle}
      </text>
      <text x={12} y={(m.t + H - m.b) / 2} text-anchor="middle" class="fill-muted" font-size={compact ? 10.5 : 12} transform={`rotate(-90 12 ${(m.t + H - m.b) / 2})`}>
        {yTitle}
      </text>
      {labels.map((l) => (l.lead ? <line x1={l.px} y1={l.py} x2={l.ex} y2={l.ey} class="stroke-line-strong" stroke-width="1" /> : null))}
      {pts.map(({ p, px, py }) => (
        <g>
          <title>{`${p.label}: ${fmt(p.y)}, ${Math.round(p.x)} ms`}</title>
          <MarkSvg cx={px} cy={py} r={compact ? 5 : 6} tone={p.tone} mark={p.mark} />
        </g>
      ))}
      {labels.map((l) => (
        <text
          x={l.x}
          y={l.y}
          font-size={fs}
          class={l.p.tone === 'us' ? 'fill-fg' : undefined}
          style={l.p.tone === 'us' ? undefined : `fill:${toneVar(l.p.tone)}`}
          font-family="ui-monospace, SFMono-Regular, Menlo, monospace"
        >
          {l.p.label}
        </text>
      ))}
    </svg>
  )
}

// --------------------------------------------------------------------------------------------
// Horizontal bars with values (HTML, like the home page's scoreboard)

export interface BarRow {
  label: string
  value: number
  text: string
  tone: Tone
  note?: string
}

export function Bars({ rows, max, label }: { rows: BarRow[]; max: number; label: string }) {
  return (
    <div class="grid grid-cols-[8.5rem_minmax(0,1fr)] gap-x-3 sm:grid-cols-[13rem_minmax(0,1fr)] sm:gap-x-4" role="list" aria-label={label}>
      {rows.map((r) => (
        <div class="contents" role="listitem">
          <span class="flex min-h-9 flex-col justify-center py-0.5 leading-tight">
            <span class="font-mono text-xs [overflow-wrap:anywhere] text-fg sm:text-[13px]">{r.label}</span>
            {r.note ? <span class="text-[11px] text-muted sm:text-xs">{r.note}</span> : null}
          </span>
          <span class="flex min-h-9 items-center">
            <span class="min-w-0 flex-1">
              <span
                class="block h-2.5 min-w-1 rounded-full"
                style={`width:${Math.max(0.5, Math.min(100, (r.value / max) * 100)).toFixed(1)}%;background:${toneVar(r.tone)}`}
              ></span>
            </span>
            <span class="ml-2.5 w-16 text-[13px] font-medium whitespace-nowrap text-fg tabular-nums">{r.text}</span>
          </span>
        </div>
      ))}
    </div>
  )
}

// --------------------------------------------------------------------------------------------
// Heatmap: one row per model, one column per dataset

export function Heatmap({
  rows,
  columns,
  lo,
  hi,
  label,
}: {
  rows: { label: string; tone?: Tone; values: (number | null)[] }[]
  columns: { key: string; label: string }[]
  lo: number
  hi: number
  label: string
}) {
  // A single-hue ramp of the Ollaya blue: lighter is lower, darker is higher.
  const shade = (v: number) => Math.round(Math.max(0, Math.min(1, (v - lo) / (hi - lo))) * 88 + 6)
  return (
    <div class="overflow-x-auto">
      <table class="w-full border-separate border-spacing-0.5 text-center text-[11px] tabular-nums" aria-label={label}>
        <thead>
          <tr>
            <th scope="col" class="sticky left-0 z-10 bg-canvas"></th>
            {columns.map((c) => (
              <th scope="col" class="h-24 min-w-9 align-bottom font-normal text-muted">
                <span class="inline-block rotate-180 whitespace-nowrap [writing-mode:vertical-rl]">{c.label}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr>
              <th
                scope="row"
                class="sticky left-0 z-10 bg-canvas pr-2 text-left font-mono text-[11px] font-normal sm:text-xs sm:whitespace-nowrap"
                style={r.tone && r.tone !== 'us' ? `color:${toneVar(r.tone)}` : undefined}
              >
                {r.tone && r.tone !== 'us' ? r.label : <span class="text-fg">{r.label}</span>}
              </th>
              {r.values.map((v) => {
                if (v === null) return <td class="rounded-sm bg-fill text-faint">·</td>
                const s = shade(v)
                return (
                  <td
                    class={`h-8 rounded-sm ${s > 55 ? 'text-white' : 'text-fg'}`}
                    style={`background:color-mix(in oklab, var(--color-us) ${s}%, var(--color-canvas))`}
                    title={`${r.label}: ${fmt(v)}`}
                  >
                    {Math.round(v * 100)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// --------------------------------------------------------------------------------------------
// Dot plot: one row per model, one dot per device, on a log axis of milliseconds

export interface DotSeries extends SeriesKey {
  key: string
}

export function DotPlot({
  rows,
  series,
  ticks,
  label,
  standalone,
  compact,
}: {
  rows: { label: string; values: Record<string, number | undefined> }[]
  series: DotSeries[]
  ticks: number[]
  label: string
  /** Render as its own .svg file, with its colours and the legend inside. */
  standalone?: boolean
  /** The phone layout: a narrower canvas, so the text keeps its size on a small screen. */
  compact?: boolean
}) {
  const W = compact ? 400 : 760
  const rowH = compact ? 22 : 24
  // Wide enough for the longest model name (a monospace glyph is about 0.61 em).
  const labelW = Math.max(0, ...rows.map((r) => r.label.length)) * (compact ? 10 : 11) * 0.61
  const m = { l: Math.max(compact ? 122 : 150, Math.ceil(labelW) + 12), r: compact ? 14 : 16, t: standalone ? 34 : 8, b: 34 }
  const shown = compact ? ticks.filter((t) => [10, 100, 1000, 10000, 100000].includes(t)) : ticks
  const H = m.t + rows.length * rowH + m.b
  const x = logScale(ticks[0]!, ticks.at(-1)!, m.l, W - m.r)
  const sa = standaloneParts(W, H, standalone)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} class="h-auto w-full" role="img" aria-label={label} {...sa.attrs}>
      <title>{label}</title>
      {sa.head}
      {standalone ? <SvgLegend items={series} x={m.l} y={14} /> : null}
      {shown.map((t) => (
        <g>
          <line x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} class="stroke-line" stroke-width="1" />
          <text x={x(t)} y={H - m.b + 16} text-anchor="middle" class="fill-muted" font-size={compact ? 10 : 11}>
            {tickLabel(t)}
          </text>
        </g>
      ))}
      <text x={(m.l + W - m.r) / 2} y={H - 4} text-anchor="middle" class="fill-muted" font-size="11">
        milliseconds for five questions, log scale
      </text>
      {rows.map((r, i) => {
        const cy = m.t + i * rowH + rowH / 2
        const vals = series.map((s) => r.values[s.key]).filter((v): v is number => v !== undefined)
        const lo = vals.length ? Math.min(...vals) : 0
        const hi = vals.length ? Math.max(...vals) : 0
        return (
          <g>
            <text x={m.l - 8} y={cy + 4} text-anchor="end" font-size={compact ? 10 : 11} class="fill-fg" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">
              {r.label}
            </text>
            {vals.length > 1 ? <line x1={x(lo)} x2={x(hi)} y1={cy} y2={cy} class="stroke-line-strong" stroke-width="2" /> : null}
            {series.map((s) => {
              const v = r.values[s.key]
              if (v === undefined) return null
              return (
                <g>
                  <title>{`${r.label} on ${s.label}: ${Math.round(v)} ms`}</title>
                  <MarkSvg cx={x(v)} cy={cy} r={compact ? 5 : 5.5} tone={s.tone} mark={s.mark} />
                </g>
              )
            })}
          </g>
        )
      })}
    </svg>
  )
}

/** A chart drawn twice: the full layout from sm up, the compact one on a phone. */
export function Responsive({ wide, compact }: { wide: Child; compact: Child }) {
  return (
    <>
      <div class="hidden sm:block">{wide}</div>
      <div class="sm:hidden">{compact}</div>
    </>
  )
}

/** A figure: the caption above, the legend, the chart, and notes under it. */
export function Figure({ title, sub, legend, children, notes }: { title: string; sub?: string; legend?: SeriesKey[]; children: Child; notes?: Child }) {
  return (
    <figure>
      <figcaption class="text-sm font-medium text-fg">
        {title}
        {sub ? <span class="font-normal text-muted"> · {sub}</span> : null}
      </figcaption>
      {legend?.length ? (
        <div class="mt-4">
          <Legend items={legend} />
        </div>
      ) : null}
      <div class="mt-5">{children}</div>
      {notes ? <div class="mt-4 max-w-3xl text-[13px] leading-relaxed text-muted">{notes}</div> : null}
    </figure>
  )
}
