import { useEffect, useRef, useState } from 'react'

const BASE = import.meta.env.BASE_URL

const ARTIFACTS = [
  {
    slug: 'ai-systems-notes',
    title: 'OS Fundamentals for AI Systems',
    kind: 'Notes · OS',
    description: 'Four interactive modules on the host side of distributed training: machine anatomy, virtual memory, CPU scheduling, and I/O + networking + containers.',
    tags: ['OS', 'Systems', 'Distributed Training', 'CUDA'],
  },
  {
    slug: 'pytorch-allocator-map',
    title: 'PyTorch Allocator Map',
    kind: 'Reference · PyTorch',
    description: "A reference map of the PyTorch CUDA memory allocator: block states, pool kinds, segment lifecycle, and the caching allocator's internal bookkeeping.",
    tags: ['PyTorch', 'CUDA', 'Memory'],
  },
  {
    slug: 'comm-compute-fusion',
    title: 'Comm–Compute Fusion Patterns',
    kind: 'Reference · Distributed Training',
    description: 'Visual patterns for overlapping communication and compute in distributed training: ring-allreduce bucketing, pipeline bubble reduction, and tensor-parallel overlap.',
    tags: ['Distributed Training', 'NCCL', 'Overlap', 'PyTorch'],
  },
]

function store(k, v) {
  try {
    if (v === undefined) return localStorage.getItem(k)
    localStorage.setItem(k, v)
  } catch { return null }
}

function applyTheme(t) {
  if (t) document.documentElement.setAttribute('data-theme', t)
  else document.documentElement.removeAttribute('data-theme')
}

export default function App() {
  const [slug, setSlug] = useState(() => location.hash.slice(1) || '')
  const [, forceRender] = useState(0)
  const frameRef = useRef(null)
  const artifact = ARTIFACTS.find(a => a.slug === slug) ?? null

  // Apply saved theme on mount
  useEffect(() => { applyTheme(store('osf:theme')) }, [])

  // Sync URL hash → state (browser back/forward)
  useEffect(() => {
    const handler = () => setSlug(location.hash.slice(1) || '')
    window.addEventListener('hashchange', handler)
    return () => window.removeEventListener('hashchange', handler)
  }, [])

  function navigate(s) {
    setSlug(s)
    location.hash = s
    window.scrollTo(0, 0)
  }

  function goHome(e) {
    e.preventDefault()
    setSlug('')
    history.replaceState(null, '', location.pathname + location.search)
  }

  function toggleTheme() {
    const root = document.documentElement
    const isDark =
      root.getAttribute('data-theme') === 'dark' ||
      (!root.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches)
    const next = isDark ? 'light' : 'dark'
    applyTheme(next)
    store('osf:theme', next)
    forceRender(n => n + 1)
    // Push theme into the iframe so it updates immediately
    try { frameRef.current?.contentWindow?.postMessage({ type: 'ca-theme', value: next }, '*') }
    catch { /* cross-origin guard */ }
  }

  return (
    <>
      <div className="top">
        <div className="in">
          {artifact ? (
            <>
              <a href={location.pathname} className="home-link" onClick={goHome}>← claude·artifacts</a>
              <span className="sep-dot">·</span>
              <span className="page-title">{artifact.title}</span>
            </>
          ) : (
            <span className="brand">claude<span>·artifacts</span></span>
          )}
          <span className="sp" />
          <button className="btn" onClick={toggleTheme} title="Toggle light/dark">◐ Theme</button>
        </div>
      </div>

      {artifact ? (
        <iframe
          ref={frameRef}
          key={slug}
          src={`${BASE}${slug}/`}
          className="article-frame"
          title={artifact.title}
        />
      ) : (
        <Landing onSelect={navigate} />
      )}
    </>
  )
}

function Landing({ onSelect }) {
  return (
    <>
      <div className="hero">
        <div className="kicker">Research notes and tools</div>
        <h1>AI systems artifacts</h1>
        <p className="lede">
          Interactive references, notes, and write-ups from explorations in GPU programming,
          PyTorch internals, and distributed training.
        </p>
      </div>
      <div className="grid">
        {ARTIFACTS.map(a => (
          <a
            key={a.slug}
            className="card"
            href={`#${a.slug}`}
            onClick={e => { e.preventDefault(); onSelect(a.slug) }}
          >
            <div className="n">{a.kind}</div>
            <h3>{a.title}</h3>
            <p>{a.description}</p>
            <div className="tags">
              {a.tags.map(t => <span key={t} className="tag">{t}</span>)}
            </div>
          </a>
        ))}
      </div>
      <footer className="foot">
        <a href="https://kapilsharma.dev" target="_blank" rel="noopener noreferrer">kapilsharma.dev</a>
        <span className="sep">·</span>
        <a href="https://github.com/kapilsh/claude-artifacts" target="_blank" rel="noopener noreferrer">GitHub</a>
      </footer>
    </>
  )
}
