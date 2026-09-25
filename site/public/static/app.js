// Ollaya site — small progressive enhancements. No dependencies.
// Every page works without this file; it adds search filtering, the navbar typeahead,
// tabs, copy buttons, relative dates, the mobile menu scroll lock and the home page rail.
;(() => {
  'use strict'

  const $ = (sel, root = document) => root.querySelector(sel)
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel))
  const isTyping = (el) => !!el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))
  const isVisible = (el) => !!el && el.getClientRects().length > 0
  const normalize = (s) => s.toLowerCase().normalize('NFKD')
  const termsOf = (q) => normalize(q).split(/\s+/).filter(Boolean)
  const debounce = (fn, ms) => {
    let t
    return (...args) => {
      clearTimeout(t)
      t = setTimeout(() => fn(...args), ms)
    }
  }
  const el = (tag, cls, text) => {
    const node = document.createElement(tag)
    if (cls) node.className = cls
    if (text != null) node.textContent = text
    return node
  }
  const ARROW_SVG =
    '<svg class="size-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true" focusable="false"><path stroke-linecap="round" stroke-linejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3"></path></svg>'

  // --- Relative dates: <time datetime data-relative>Sep 23, 2026</time> → "today" -------------
  const DAY = 86400000
  const relative = (iso) => {
    const days = Math.floor((Date.now() - Date.parse(iso)) / DAY)
    if (!Number.isFinite(days) || days <= 0) return 'today'
    if (days === 1) return 'yesterday'
    if (days < 7) return `${days} days ago`
    const n = (v, unit) => `${v} ${unit}${v === 1 ? '' : 's'} ago`
    if (days < 30) return n(Math.floor(days / 7), 'week')
    if (days < 365) return n(Math.floor(days / 30), 'month')
    return n(Math.floor(days / 365), 'year')
  }
  $$('time[data-relative]').forEach((t) => (t.textContent = relative(t.dateTime)))

  // --- Keyboard shortcuts: ⌘K / Ctrl+K and "/" focus the model search ------------------------
  const isApple = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)
  if (!isApple) $$('[data-shortcut-hint]').forEach((n) => (n.textContent = 'Ctrl K'))

  document.addEventListener('keydown', (e) => {
    const cmdK = e.key.toLowerCase() === 'k' && (e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey
    const slash = e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey && !isTyping(document.activeElement)
    if (!cmdK && !slash) return
    const input = [$('#search-q'), $('#nav-search')].find(isVisible)
    e.preventDefault()
    if (input) {
      input.focus()
      input.select()
    } else {
      window.location.href = '/search'
    }
  })

  // --- Search index (/search.json), loaded on first use --------------------------------------
  let indexPromise
  const loadIndex = () =>
    (indexPromise ??= fetch('/search.json')
      .then((r) => (r.ok ? r.json() : { models: [] }))
      .catch(() => ({ models: [] })))

  const searchUrl = (q, caps = [], sort = 'popular') => {
    const p = new URLSearchParams()
    if (q.trim()) p.set('q', q.trim())
    caps.forEach((c) => p.append('c', c))
    if (sort !== 'popular') p.set('o', sort)
    const s = p.toString()
    return s ? `/search?${s}` : '/search'
  }

  // --- /search: filter, sort and sync the URL ------------------------------------------------
  const form = $('[data-search-form]')
  if (form) {
    const input = form.elements.namedItem('q')
    const sort = form.elements.namedItem('o')
    const checks = $$('input[name="c"]', form)
    const list = $('[data-results-list]')
    const rows = $$('li[data-search]', list)
    const empty = $('[data-results-empty]')
    const count = $('[data-results-count]')
    const queryLabel = $('[data-results-query]')

    const fromUrl = () => {
      const p = new URLSearchParams(location.search)
      input.value = p.get('q') ?? ''
      const caps = p.getAll('c')
      checks.forEach((c) => (c.checked = caps.includes(c.value)))
      sort.value = p.get('o') === 'newest' ? 'newest' : 'popular'
    }

    const apply = () => {
      const q = input.value
      const terms = termsOf(q)
      const caps = checks.filter((c) => c.checked).map((c) => c.value)
      let visible = 0
      rows.forEach((r) => {
        const rowCaps = r.dataset.caps.split(' ')
        const match = caps.every((c) => rowCaps.includes(c)) && terms.every((t) => r.dataset.search.includes(t))
        r.hidden = !match
        if (match) visible++
      })
      const byRank = (a, b) => Number(a.dataset.rank) - Number(b.dataset.rank)
      const order =
        sort.value === 'newest'
          ? (a, b) => b.dataset.updated.localeCompare(a.dataset.updated) || byRank(a, b)
          : byRank
      rows.slice().sort(order).forEach((r) => list.appendChild(r))
      list.hidden = visible === 0
      empty.hidden = visible !== 0
      queryLabel.textContent = q.trim() ? ` for “${q.trim()}”` : ''
      count.textContent = `${visible} ${visible === 1 ? 'model' : 'models'} found`
      const url = searchUrl(q, caps, sort.value).replace(/^\/search/, location.pathname)
      if (url !== location.pathname + location.search) history.replaceState(null, '', url)
    }

    fromUrl()
    apply()
    input.addEventListener('input', debounce(apply, 100))
    form.addEventListener('change', (e) => {
      if (e.target !== input) apply()
    })
    form.addEventListener('submit', (e) => {
      e.preventDefault()
      apply()
    })
    $('[data-results-clear]')?.addEventListener('click', (e) => {
      e.preventDefault()
      form.reset()
      sort.value = 'popular'
      apply()
      input.focus()
    })
  }

  // --- Navbar typeahead -----------------------------------------------------------------------
  const navForm = $('[data-typeahead]')
  const navInput = $('#nav-search')
  const navBox = $('#nav-typeahead')
  if (navForm && navInput && navBox) {
    const items = () => $$('[data-typeahead-item]', navBox)
    const close = () => (navBox.hidden = true)

    const suggest = (index, q, limit = 6) => {
      const query = normalize(q.trim())
      const terms = termsOf(q)
      const out = []
      for (const m of index.models) {
        if (!terms.every((t) => m.search.includes(t))) continue
        out.push({ href: m.href, name: m.name, detail: m.description })
        if (!query) continue
        for (const t of m.tags) if (t.search.includes(query)) out.push({ href: t.href, name: t.name, detail: t.summary })
      }
      return out.slice(0, limit)
    }

    const render = (q, results) => {
      const box = el('div')
      if (results.length) {
        const ul = el('ul', 'max-h-[60vh] overflow-y-auto py-2')
        ul.setAttribute('role', 'list')
        if (!q.trim()) {
          ul.appendChild(el('li', 'px-4 pt-1 pb-2 text-[11px] font-medium tracking-wider text-muted uppercase', 'Models'))
        }
        for (const r of results) {
          const li = el('li')
          const a = el('a', 'block px-4 py-2.5 outline-none hover:bg-fill focus-visible:bg-fill')
          a.href = r.href
          a.dataset.typeaheadItem = ''
          a.append(el('span', 'block text-sm font-medium text-fg', r.name), el('span', 'mt-0.5 block truncate text-[13px] text-muted', r.detail))
          li.appendChild(a)
          ul.appendChild(li)
        }
        box.appendChild(ul)
      } else {
        box.appendChild(el('p', 'px-4 py-4 text-sm text-muted', `No models match “${q.trim()}”.`))
      }
      const all = el(
        'a',
        'flex items-center justify-between border-t border-line px-4 py-3 text-[13px] text-muted outline-none hover:bg-fill hover:text-fg focus-visible:bg-fill',
        q.trim() ? `View all results for “${q.trim()}”` : 'Browse all models',
      )
      all.href = searchUrl(q)
      all.dataset.typeaheadItem = ''
      all.insertAdjacentHTML('beforeend', ARROW_SVG)
      box.appendChild(all)
      navBox.replaceChildren(box)
    }

    const update = async () => {
      const q = navInput.value
      const index = await loadIndex()
      if (q !== navInput.value || !navForm.contains(document.activeElement)) return
      render(q, suggest(index, q))
      navBox.hidden = false
    }

    navInput.addEventListener('focus', update)
    navInput.addEventListener('input', debounce(update, 100))
    navForm.addEventListener('focusout', (e) => {
      if (!navForm.contains(e.relatedTarget)) close()
    })
    navForm.addEventListener('keydown', (e) => {
      const list = items()
      const i = list.indexOf(document.activeElement)
      if (e.key === 'Escape') {
        close()
        navInput.blur()
      } else if (e.key === 'ArrowDown' && list.length) {
        e.preventDefault()
        navBox.hidden = false
        list[Math.min(i + 1, list.length - 1)].focus()
      } else if (e.key === 'ArrowUp' && i >= 0) {
        e.preventDefault()
        if (i === 0) navInput.focus()
        else list[i - 1].focus()
      }
    })
  }

  // --- Tabs (WAI-ARIA tabs pattern) -----------------------------------------------------------
  $$('[role="tablist"]').forEach((list) => {
    const tabs = $$('[role="tab"]', list)
    const select = (tab, focus) => {
      tabs.forEach((t) => {
        const on = t === tab
        t.setAttribute('aria-selected', on ? 'true' : 'false')
        t.tabIndex = on ? 0 : -1
        const panel = document.getElementById(t.getAttribute('aria-controls'))
        if (panel) panel.hidden = !on
      })
      if (focus) tab.focus()
    }
    tabs.forEach((tab, i) => {
      tab.addEventListener('click', () => select(tab, false))
      tab.addEventListener('keydown', (e) => {
        const n = tabs.length
        const to =
          e.key === 'ArrowRight' ? tabs[(i + 1) % n]
          : e.key === 'ArrowLeft' ? tabs[(i - 1 + n) % n]
          : e.key === 'Home' ? tabs[0]
          : e.key === 'End' ? tabs[n - 1]
          : null
        if (to) {
          e.preventDefault()
          select(to, true)
        }
      })
    })
    // Download page: preselect the visitor's platform (Linux is the default).
    if (list.hasAttribute('data-os-tabs')) {
      const ua = navigator.userAgent
      const os = /iPhone|iPad/.test(ua) ? null : /Macintosh|Mac OS X/.test(ua) ? 'macos' : /Windows/.test(ua) ? 'windows' : null
      const tab = os && tabs.find((t) => t.id === `os-tab-${os}`)
      if (tab) select(tab, false)
    }
  })

  // --- Copy buttons ---------------------------------------------------------------------------
  const copyText = async (text) => {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      const ta = document.createElement('textarea')
      ta.value = text
      ta.setAttribute('readonly', '')
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      let ok = false
      try {
        ok = document.execCommand('copy')
      } catch {}
      ta.remove()
      return ok
    }
  }

  document.addEventListener('click', async (e) => {
    const btn = e.target instanceof Element ? e.target.closest('[data-copy]') : null
    if (!btn) return
    const scope = btn.closest('[data-copy-scope]') || document
    const pre = $$('[role="tabpanel"]', scope).find((p) => !p.hidden)?.querySelector('pre') || $('pre', scope)
    if (!pre) return
    const ok = await copyText(pre.innerText.replace(/\n$/, ''))
    const status = $('[data-copy-status]', btn)
    if (!ok) return
    btn.setAttribute('data-copied', '')
    if (status) status.textContent = 'Copied'
    clearTimeout(btn._copyTimer)
    btn._copyTimer = setTimeout(() => {
      btn.removeAttribute('data-copied')
      if (status) status.textContent = ''
    }, 2000)
  })

  // --- Mobile menu: lock page scroll while open, close on Escape ------------------------------
  const menu = $('[data-mobile-menu]')
  if (menu) {
    menu.addEventListener('toggle', () => {
      document.documentElement.classList.toggle('overflow-hidden', menu.open)
    })
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && menu.open) {
        menu.open = false
        $('summary', menu)?.focus()
      }
    })
    window.matchMedia('(min-width: 768px)').addEventListener('change', (e) => {
      if (e.matches) menu.open = false
    })
  }

  // --- Home page section rail: highlight the section in view ----------------------------------
  const rail = $('[data-scrollspy]')
  if (rail && 'IntersectionObserver' in window) {
    const links = $$('a[href^="#"]', rail)
    const byId = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]))
    const setActive = (id) =>
      links.forEach((a) => {
        if (a === byId.get(id)) a.setAttribute('aria-current', 'true')
        else a.removeAttribute('aria-current')
      })
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) setActive(entry.target.id)
        })
      },
      { rootMargin: '-35% 0px -60% 0px' },
    )
    byId.forEach((_, id) => {
      const section = document.getElementById(id)
      if (section) observer.observe(section)
    })
  }
})()
