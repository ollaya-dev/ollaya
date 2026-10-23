// The design demo's charts (src/pages/design.tsx): a tooltip on each point, by pointer or keyboard,
// and legend buttons that show or hide a series. The SVG is complete without this script.
;(() => {
  for (const root of document.querySelectorAll('[data-chart]')) {
    const tip = root.querySelector('[data-tip]')
    const svg = root.querySelector('svg[role="img"]')
    if (!tip || !svg) continue

    // The same model elsewhere on the chart (another tag, or the same weights on the other server) lights up with
    // a line to it, so Ollaya and Ollama can be read pair by pair.
    const links = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    links.setAttribute('class', 'links')
    svg.appendChild(links)
    function show(pt) {
      const d = pt.dataset
      const kin = [...svg.querySelectorAll(`.pt[data-family="${d.family}"]`)].filter((p) => p !== pt && !p.classList.contains('is-off'))
      links.replaceChildren(
        ...kin.map((k) => {
          const l = document.createElementNS('http://www.w3.org/2000/svg', 'line')
          l.setAttribute('x1', d.x)
          l.setAttribute('y1', d.y)
          l.setAttribute('x2', k.dataset.x)
          l.setAttribute('y2', k.dataset.y)
          return l
        }),
      )
      for (const k of kin) k.classList.add('is-kin')
      tip.innerHTML = `<b>${d.name}</b> <span>on ${d.server}</span><br>accuracy ${d.acc} · ECE ${d.ece} · ${d.ms} ms${kin.length ? `<br><span>${kin.map((k) => `${k.dataset.name} on ${k.dataset.server}: ${k.dataset.acc}, ${k.dataset.ms} ms`).join(' · ')}</span>` : ''}`
      tip.hidden = false
      const r = pt.getBoundingClientRect()
      const b = root.getBoundingClientRect()
      const x = r.left - b.left + r.width / 2
      const y = r.top - b.top
      const w = tip.offsetWidth
      tip.style.left = `${Math.max(8, Math.min(b.width - w - 8, x - w / 2))}px`
      tip.style.top = `${y - tip.offsetHeight - 10}px`
      for (const p of svg.querySelectorAll('.pt')) p.classList.toggle('is-on', p === pt)
    }
    function hide() {
      tip.hidden = true
      links.replaceChildren()
      for (const p of svg.querySelectorAll('.pt')) p.classList.remove('is-on', 'is-kin')
    }
    for (const pt of svg.querySelectorAll('.pt')) {
      pt.addEventListener('pointerenter', () => show(pt))
      pt.addEventListener('focus', () => show(pt))
      pt.addEventListener('pointerleave', hide)
      pt.addEventListener('blur', hide)
    }
    for (const btn of root.querySelectorAll('[data-series]')) {
      btn.addEventListener('click', () => {
        const on = btn.getAttribute('aria-pressed') !== 'true'
        btn.setAttribute('aria-pressed', String(on))
        for (const p of svg.querySelectorAll(`.pt[data-server="${btn.dataset.series}"]`)) p.classList.toggle('is-off', !on)
        hide()
      })
    }
  }
})()
