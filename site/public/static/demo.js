// The hero's example (src/pages/design.tsx): a button cycles through the recorded examples, another
// shows every option of every question instead of the top one. Without the script the first example
// shows, folded.
;(() => {
  const root = document.querySelector('[data-demo]')
  if (!root) return
  const panels = () => [...root.querySelectorAll('[data-example]')]
  let at = 0
  root.querySelector('[data-next]')?.addEventListener('click', () => {
    const all = panels()
    all[at].hidden = true
    at = (at + 1) % all.length
    // A fresh copy of the node, so its entrance animation runs again.
    const fresh = all[at].cloneNode(true)
    fresh.hidden = false
    all[at].replaceWith(fresh)
  })
  const toggle = root.querySelector('[data-toggle]')
  toggle?.addEventListener('click', () => {
    const open = !root.classList.contains('is-open')
    root.classList.toggle('is-open', open)
    root.classList.add('was-open')
    toggle.setAttribute('aria-pressed', String(open))
    toggle.textContent = open ? 'Top option only' : 'Show every option'
  })
})()
