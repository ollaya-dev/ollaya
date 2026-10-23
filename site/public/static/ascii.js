// A procedural ASCII field behind the hero (src/pages/design.tsx): a slow value field mapped onto a
// character ramp, drawn on a canvas in the page's data colours at low opacity. The cursor sends a
// ripple through the characters and lights the field around it. Reduced motion draws one still frame.
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
  let W, H, cols, rows
  let mx = -1e9
  let my = -1e9

  function size() {
    const dpr = Math.min(devicePixelRatio || 1, 2)
    W = host.clientWidth
    H = host.clientHeight
    canvas.width = W * dpr
    canvas.height = H * dpr
    canvas.style.width = `${W}px`
    canvas.style.height = `${H}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.font = `${fs}px ui-monospace, SFMono-Regular, Menlo, monospace`
    ctx.textBaseline = 'top'
    cols = Math.ceil(W / cw)
    rows = Math.ceil(H / ch)
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

  function draw(t) {
    ctx.clearRect(0, 0, W, H)
    const n = ramp.length - 1
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const d = Math.hypot(c - mx, (r - my) * 1.9)
        const ripple = d < 9 ? Math.cos(d * 0.7 - t * 3) * (1 - d / 9) * 0.9 : 0
        const v = Math.max(-1, Math.min(1, field(c, r, t) + ripple))
        const i = Math.round(((v + 1) / 2) * n)
        if (i < 9) continue // the sparse end of the ramp stays empty, so the field has clearings
        ctx.globalAlpha = base + (d < 11 ? (0.85 - base) * (1 - d / 11) : 0)
        ctx.fillStyle = palette[Math.floor((c / cols + (v + 1) / 2) * 2) % palette.length]
        ctx.fillText(ramp[i], c * cw, r * ch)
      }
    }
  }

  size()
  addEventListener('resize', size)
  if (still) return draw(0)
  let visible = true
  new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(host)
  let last = 0
  function frame(now) {
    if (visible && now - last > 60) {
      last = now
      draw(now / 1000)
    }
    requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
})()
