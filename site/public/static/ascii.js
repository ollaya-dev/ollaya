// A sparse character field behind the hero (src/pages/home.tsx): a slow value field decides, cell by
// cell, whether a letter of the page's words, a plus, a colon or nothing is drawn. Most cells stay
// empty, so the field reads as texture at the edges and leaves the middle to the headline (the
// .ascii element's mask clears the centre). One accent colour for the letters, ink for the rest.
// The cursor lights the cells around it. Reduced motion draws one still frame.
//
// Cheap on purpose: every glyph is drawn once into an atlas and stamped with drawImage; the whole
// field is redrawn a few times a second and only the cells around the cursor between those frames.
;(() => {
  const host = document.querySelector('[data-ascii]')
  if (!host) return
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches
  const canvas = document.createElement('canvas')
  host.appendChild(canvas)
  const ctx = canvas.getContext('2d')
  const words = 'DECISION MODELS LOCALLY  OLLAYA  ROUTING GUARDRAILS TRIAGE  '
  // Glyph slots: the words' letters, then the fill marks.
  const glyphs = [...new Set(words.replace(/ /g, ''))].join('') + '+:.'
  const PLUS = glyphs.length - 3
  const COLON = glyphs.length - 2
  const DOT = glyphs.length - 1
  const inks = ['#8fd3ae', '#7a7a7a'] // letters, marks
  const fs = 11
  const cw = fs * 0.95
  const ch = fs * 1.35
  const R = 9 // the cursor's reach, in cells
  let dpr, W, H, cols, rows, atlas, aw, ah
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
    aw = Math.ceil(cw * dpr)
    ah = Math.ceil(ch * dpr)
    atlas = document.createElement('canvas')
    atlas.width = aw * glyphs.length
    atlas.height = ah * inks.length
    const a = atlas.getContext('2d')
    a.setTransform(dpr, 0, 0, dpr, 0, 0)
    a.font = `500 ${fs}px ui-monospace, SFMono-Regular, Menlo, monospace`
    a.textBaseline = 'top'
    a.textAlign = 'center'
    inks.forEach((colour, p) => {
      a.fillStyle = colour
      for (let i = 0; i < glyphs.length; i++) a.fillText(glyphs[i], (i * aw) / dpr + cw / 2, (p * ah) / dpr + 1)
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

  // Broad, slow bands: the field changes over tens of cells, not from one cell to the next.
  function field(x, y, t) {
    return (
      (Math.sin(x * 0.045 + t * 0.18) +
        Math.sin(y * 0.11 - t * 0.12) +
        Math.sin((x - y * 1.6) * 0.03 + t * 0.09) +
        Math.sin(Math.hypot(x - cols * 0.5, (y - rows * 0.45) * 1.4) * 0.05 - t * 0.2)) /
      4
    )
  }

  function cell(c, r, t) {
    const d = Math.hypot(c - mx, (r - my) * 1.4)
    const near = d < R ? 1 - d / R : 0
    const v = field(c, r, t) + near * 0.35
    let g, ink, alpha
    if (v > 0.5) {
      // A letter of the words, read along the row so neighbouring cells spell them.
      const letter = words[(c + r * 7) % words.length]
      if (letter === ' ') return
      g = glyphs.indexOf(letter)
      ink = 0
      alpha = 0.16
    } else if (v > 0.28) {
      g = (c + r) % 3 === 0 ? PLUS : COLON
      ink = 1
      alpha = 0.22
    } else if (v > 0.1) {
      g = DOT
      ink = 1
      alpha = 0.22
    } else return
    ctx.globalAlpha = Math.min(1, alpha + near * 0.6)
    ctx.drawImage(atlas, g * aw, ink * ah, aw, ah, c * cw, r * ch, cw, ch)
  }

  /** The cells the cursor reaches, as a box to clear and redraw. */
  function boxAt() {
    if (mx < -1e8) return null
    const c0 = Math.max(0, Math.floor(mx - R - 1))
    const c1 = Math.min(cols, Math.ceil(mx + R + 1))
    const r0 = Math.max(0, Math.floor(my - R / 1.4 - 1))
    const r1 = Math.min(rows, Math.ceil(my + R / 1.4 + 1))
    return { x: c0 * cw, y: r0 * ch, w: (c1 - c0) * cw, h: (r1 - r0) * ch, c0, c1, r0, r1 }
  }
  function drawBox(b, t) {
    ctx.clearRect(b.x, b.y, b.w, b.h)
    for (let r = b.r0; r < b.r1; r++) for (let c = b.c0; c < b.c1; c++) cell(c, r, t)
  }

  // Wherever the cursor last lit the field is remembered, so the next frame puts it back.
  let lit = null
  function drawAll(t) {
    ctx.clearRect(0, 0, W, H)
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) cell(c, r, t)
    lit = boxAt()
  }
  function drawNear(t) {
    const b = boxAt()
    if (lit) drawBox(lit, t)
    if (b) drawBox(b, t)
    lit = b
  }

  size()
  // The hero grows when its example opens every option; the field follows its host's size.
  new ResizeObserver(() => {
    if (host.clientWidth === W && host.clientHeight === H) return
    size()
    drawAll(performance.now() / 1000)
  }).observe(host)
  if (still) return drawAll(0)
  let visible = true
  new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(host)
  let lastAll = 0
  let lastNear = 0
  function frame(now) {
    if (visible) {
      const t = now / 1000
      if (now - lastAll > 200) {
        lastAll = now
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
