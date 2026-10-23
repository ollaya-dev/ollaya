// A procedural ASCII field behind the hero (src/pages/design.tsx): a slow value field mapped onto a
// character ramp, drawn on a canvas in the page's data colours at low opacity. The cursor sends a
// ripple through the characters and lights the field around it. Reduced motion draws one still frame.
//
// Cheap on purpose: every glyph is drawn once into an atlas and stamped with drawImage; the whole
// field is redrawn eight times a second (it drifts slowly) and only the cells around the cursor are
// redrawn between those frames.
;(() => {
  const host = document.querySelector('[data-ascii]')
  if (!host) return
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches
  const canvas = document.createElement('canvas')
  host.appendChild(canvas)
  const ctx = canvas.getContext('2d')
  const ramp = " .`',:;!i<>+~-_?][}{)(|/tfjrxnuvczXYUJCLQ0"
  const palette = ['#8cc8ff', '#9fe3b4', '#ffd080', '#d2a8ff']
  const fs = 12
  const cw = fs * 0.62
  const ch = fs * 1.2
  const base = 0.13
  const SPARSE = 9 // the sparse end of the ramp stays empty, so the field has clearings
  const R = 11 // the cursor's reach, in cells
  let dpr, W, H, cols, rows, atlas
  let mx = -1e9
  let my = -1e9

  function size() {
    dpr = Math.min(devicePixelRatio || 1, 2)
    W = host.clientWidth
    H = host.clientHeight
    canvas.width = W * dpr
    canvas.height = H * dpr
    canvas.style.width = `${W}px`
    canvas.style.height = `${H}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    cols = Math.ceil(W / cw)
    rows = Math.ceil(H / ch)
    // The atlas: one row per colour, one cell per ramp character, at the device's resolution.
    atlas = document.createElement('canvas')
    atlas.width = Math.ceil(cw * dpr) * ramp.length
    atlas.height = Math.ceil(ch * dpr) * palette.length
    const a = atlas.getContext('2d')
    a.setTransform(dpr, 0, 0, dpr, 0, 0)
    a.font = `${fs}px ui-monospace, SFMono-Regular, Menlo, monospace`
    a.textBaseline = 'top'
    palette.forEach((colour, p) => {
      a.fillStyle = colour
      for (let i = 0; i < ramp.length; i++) a.fillText(ramp[i], i * Math.ceil(cw * dpr) / dpr, p * Math.ceil(ch * dpr) / dpr)
    })
  }

  const area = host.parentElement
  area.addEventListener('pointermove', (e) => {
    const r = host.getBoundingClientRect()
    mx = (e.clientX - r.left) / cw
    my = (e.clientY - r.top) / ch
  })
  area.addEventListener('pointerleave', () => {
    mx = my = -1e9
  })

  function field(x, y, t) {
    return (
      (Math.sin(x * 0.09 + t * 0.5) +
        Math.sin(y * 0.14 - t * 0.35) +
        Math.sin((x + y) * 0.05 + t * 0.25) +
        Math.sin(Math.hypot(x - cols * 0.62, y - rows * 0.38) * 0.11 - t * 0.7)) /
      4
    )
  }

  const n = ramp.length - 1
  const aw = Math.ceil(cw * dpr)
  const ah = Math.ceil(ch * dpr)
  function cell(c, r, t) {
    const d = Math.hypot(c - mx, (r - my) * 1.9)
    const ripple = d < R ? Math.cos(d * 0.7 - t * 3) * (1 - d / R) * 0.9 : 0
    const v = Math.max(-1, Math.min(1, field(c, r, t) + ripple))
    const i = Math.round(((v + 1) / 2) * n)
    if (i < SPARSE) return
    ctx.globalAlpha = base + (d < R ? (0.85 - base) * (1 - d / R) : 0)
    const p = Math.floor((c / cols + (v + 1) / 2) * 2) % palette.length
    ctx.drawImage(atlas, i * aw, p * ah, aw, ah, c * cw, r * ch, cw, ch)
  }

  function drawAll(t) {
    ctx.clearRect(0, 0, W, H)
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) cell(c, r, t)
  }

  // Only the cells the cursor reaches: cleared and redrawn, so the ripple moves between full frames.
  let lastBox = null
  function drawNear(t) {
    if (lastBox) {
      ctx.clearRect(lastBox.x, lastBox.y, lastBox.w, lastBox.h)
      for (let r = lastBox.r0; r < lastBox.r1; r++) for (let c = lastBox.c0; c < lastBox.c1; c++) cell(c, r, t)
      lastBox = null
    }
    if (mx < -1e8) return
    const c0 = Math.max(0, Math.floor(mx - R - 1))
    const c1 = Math.min(cols, Math.ceil(mx + R + 1))
    const r0 = Math.max(0, Math.floor(my - R / 1.9 - 1))
    const r1 = Math.min(rows, Math.ceil(my + R / 1.9 + 1))
    lastBox = { x: c0 * cw, y: r0 * ch, w: (c1 - c0) * cw, h: (r1 - r0) * ch, c0, c1, r0, r1 }
    ctx.clearRect(lastBox.x, lastBox.y, lastBox.w, lastBox.h)
    for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) cell(c, r, t)
  }

  size()
  addEventListener('resize', () => {
    size()
    drawAll(performance.now() / 1000)
  })
  if (still) return drawAll(0)
  let visible = true
  new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(host)
  let lastAll = 0
  let lastNear = 0
  function frame(now) {
    if (visible) {
      const t = now / 1000
      if (now - lastAll > 125) {
        lastAll = now
        lastBox = null
        drawAll(t)
      } else if (now - lastNear > 33) {
        lastNear = now
        drawNear(t)
      }
    }
    requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
})()
