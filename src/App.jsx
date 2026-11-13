const BASE = import.meta.env.BASE_URL

const ARTIFACTS = [
  {
    slug: 'pytorch-allocator-map',
    title: 'PyTorch Allocator Map',
    description: "A reference map of the PyTorch CUDA memory allocator: block states, pool kinds, segment lifecycle, and the caching allocator's internal bookkeeping — annotated for navigating real allocator traces.",
    tags: ['PyTorch', 'CUDA', 'Memory'],
    kind: 'reference',
    href: `${BASE}pytorch-allocator-map/`,
  },
]

const KIND_LABEL = {
  reference: 'Reference',
  notes: 'Notes',
  tool: 'Tool',
  writeup: 'Write-up',
}

function Mark() {
  return (
    <svg className="mark" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="#0b0f0a"/>
      <rect x="8" y="6" width="14" height="18" rx="2" fill="#25301f"/>
      <rect x="10.5" y="10" width="9" height="1.5" rx="0.75" fill="#76b900"/>
      <rect x="10.5" y="13" width="6.5" height="1.5" rx="0.75" fill="#4a5e3a"/>
      <rect x="10.5" y="16" width="8" height="1.5" rx="0.75" fill="#4a5e3a"/>
      <rect x="10.5" y="19" width="5" height="1.5" rx="0.75" fill="#4a5e3a"/>
      <path d="M22 20 L23.4 23.4 L26.8 24.8 L23.4 26.2 L22 29.6 L20.6 26.2 L17.2 24.8 L20.6 23.4 Z" fill="#76b900"/>
    </svg>
  )
}

function ArtifactCard({ artifact }) {
  return (
    <a className="card" href={artifact.href}>
      <div className="card-header">
        <span className={`kind-badge kind-${artifact.kind}`}>{KIND_LABEL[artifact.kind]}</span>
      </div>
      <h2 className="card-title">{artifact.title}</h2>
      <p className="card-desc">{artifact.description}</p>
      <div className="card-tags">
        {artifact.tags.map((t) => (
          <span key={t} className="tag">{t}</span>
        ))}
      </div>
      <span className="card-arrow" aria-hidden="true">→</span>
    </a>
  )
}

export default function App() {
  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <Mark />
          claude<span>artifacts</span>
        </div>
        <p className="tagline">research notes and tools — generated from explorations in AI systems and GPU programming</p>
      </header>

      <main>
        <div className="grid">
          {ARTIFACTS.map((a) => (
            <ArtifactCard key={a.slug} artifact={a} />
          ))}
        </div>
      </main>

      <footer className="foot">
        <a href="https://kapilsharma.dev" target="_blank" rel="noopener noreferrer">kapilsharma.dev</a>
        <span className="sep">·</span>
        <a href="https://github.com/kapilsh/claude-artifacts" target="_blank" rel="noopener noreferrer">GitHub</a>
      </footer>
    </div>
  )
}
