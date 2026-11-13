import { useEffect, useState } from 'react'

const BASE = import.meta.env.BASE_URL

const ARTIFACTS = [
  {
    slug: 'ai-systems-notes',
    kind: 'NOTES · OS FUNDAMENTALS',
    title: 'OS Fundamentals for AI Systems',
    description: 'Four interactive modules on the host side of distributed training: machine anatomy, virtual memory, CPU scheduling, and I/O + networking + containers. Each module has interactive steppers, a drill you run on real hardware, and a self-check.',
    tags: ['OS', 'Systems', 'Distributed Training', 'CUDA'],
    href: `${BASE}ai-systems-notes/`,
  },
  {
    slug: 'pytorch-allocator-map',
    kind: 'REFERENCE · PYTORCH INTERNALS',
    title: 'PyTorch Allocator Map',
    description: "A reference map of the PyTorch CUDA memory allocator: block states, pool kinds, segment lifecycle, and the caching allocator's internal bookkeeping — annotated for navigating real allocator traces.",
    tags: ['PyTorch', 'CUDA', 'Memory'],
    href: `${BASE}pytorch-allocator-map/`,
  },
  {
    slug: 'comm-compute-fusion',
    kind: 'REFERENCE · DISTRIBUTED TRAINING',
    title: 'Comm–Compute Fusion Patterns',
    description: 'A visual reference for overlapping communication and compute in distributed training: ring-allreduce bucketing, pipeline bubble reduction, tensor-parallel overlap, and the scheduling constraints that determine whether fusion actually helps.',
    tags: ['Distributed Training', 'NCCL', 'Overlap', 'PyTorch'],
    href: `${BASE}comm-compute-fusion/`,
  },
]

function store(k, v) {
  try {
    if (v === undefined) return localStorage.getItem(k)
    localStorage.setItem(k, v)
  } catch { return null }
}

export default function App() {
  const [_theme, setTheme] = useState(() => store('osf:theme') || '')

  useEffect(() => {
    const saved = store('osf:theme')
    if (saved) document.documentElement.setAttribute('data-theme', saved)
  }, [])

  function toggleTheme() {
    const root = document.documentElement
    const isDark = root.getAttribute('data-theme') === 'dark' ||
      (!root.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches)
    const next = isDark ? 'light' : 'dark'
    root.setAttribute('data-theme', next)
    store('osf:theme', next)
    setTheme(next)
  }

  return (
    <>
      <div className="top">
        <div className="in">
          <span className="brand">claude<span>·artifacts</span></span>
          <span className="sp" />
          <button className="btn" onClick={toggleTheme} title="Toggle light/dark">◐ Theme</button>
        </div>
      </div>

      <div className="hero">
        <div className="kicker">Research notes and tools</div>
        <h1>AI systems artifacts</h1>
        <p className="lede">
          A collection of interactive references, notes, and write-ups from explorations in GPU programming, PyTorch internals, and distributed training.
        </p>
      </div>

      <div className="grid">
        {ARTIFACTS.map((a) => (
          <a key={a.slug} className="card" href={a.href}>
            <div className="n">{a.kind}</div>
            <h3>{a.title}</h3>
            <p>{a.description}</p>
            <div className="tags">
              {a.tags.map((t) => <span key={t} className="tag">{t}</span>)}
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
