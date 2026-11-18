import { useEffect, useState } from 'react'

const BASE = import.meta.env.BASE_URL

const ARTIFACTS = [
  {
    slug: 'ai-systems-notes',
    title: 'OS Fundamentals for AI Systems',
    kind: 'Notes · OS',
    description: 'Four interactive modules on the host side of distributed training: machine anatomy, virtual memory, CPU scheduling, and I/O + networking + containers. Each module has interactive steppers, a drill you run on real hardware, and a self-check.',
    tags: ['OS', 'Systems', 'Distributed Training', 'CUDA'],
  },
  {
    slug: 'pytorch-allocator-map',
    title: 'PyTorch Allocator Map',
    kind: 'Reference · PyTorch',
    description: "A reference map of the PyTorch CUDA memory allocator: block states, pool kinds, segment lifecycle, and the caching allocator's internal bookkeeping — annotated for navigating real allocator traces.",
    tags: ['PyTorch', 'CUDA', 'Memory'],
  },
  {
    slug: 'comm-compute-fusion',
    title: 'Comm–Compute Fusion Patterns',
    kind: 'Reference · Distributed Training',
    description: 'Seven patterns for fusing GPU-to-GPU communication with computation: ring-allreduce bucketing, pipeline bubble reduction, tensor-parallel overlap, and the scheduling constraints that determine whether fusion actually helps.',
    tags: ['Distributed Training', 'NCCL', 'Overlap', 'PyTorch'],
  },
  {
    slug: 'cuda-graphs',
    title: 'PyTorch CUDA Graphs',
    kind: 'Reference · PyTorch',
    description: 'How PyTorch captures and replays CUDA graphs: the capture lifecycle, private memory pools, graph-safe RNG, make_graphed_callables, conditional nodes, and Inductor CUDA graph trees. Written from source in Simplified Technical English.',
    tags: ['PyTorch', 'CUDA', 'CUDA Graphs', 'Inductor'],
  },
  {
    slug: 'cuda-graphs-deep-dive',
    title: 'CUDA Graphs Deep Dive',
    kind: 'Reference · CUDA',
    description: 'CUDA graphs at the runtime level, from the CUDA 13.5 Programming Guide: object model, stream capture rules and modes, invalidation, update, conditional and memory nodes, device launch — plus gotchas collected from public debugging reports.',
    tags: ['CUDA', 'CUDA Graphs', 'Runtime API', 'Debugging'],
  },
]

function store(k, v) {
  try {
    if (v === undefined) return localStorage.getItem(k)
    localStorage.setItem(k, v)
  } catch { return null }
}

export default function App() {
  const [, forceRender] = useState(0)

  useEffect(() => {
    const saved = store('osf:theme')
    if (saved) document.documentElement.setAttribute('data-theme', saved)
  }, [])

  function toggleTheme() {
    const root = document.documentElement
    const isDark =
      root.getAttribute('data-theme') === 'dark' ||
      (!root.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches)
    const next = isDark ? 'light' : 'dark'
    root.setAttribute('data-theme', next)
    store('osf:theme', next)
    forceRender(n => n + 1)
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
          Interactive references, notes, and write-ups from explorations in GPU programming,
          PyTorch internals, and distributed training.
        </p>
      </div>

      <div className="grid">
        {ARTIFACTS.map(a => (
          <a key={a.slug} className="card" href={`${BASE}${a.slug}/`}>
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
