# claude-artifacts

Research notes and tools — generated from explorations in AI systems, GPU programming, and distributed training.

Deployed to [kapilsharma.dev/claude-artifacts](https://www.kapilsharma.dev/claude-artifacts/).

## Artifacts

| Artifact | Kind | Description |
|---|---|---|
| [OS Fundamentals for AI Systems](public/ai-systems-notes/) | Notes | Machine anatomy, virtual memory, CPU scheduling, I/O + networking (4 interactive modules) |
| [PyTorch Allocator Map](public/pytorch-allocator-map/index.html) | Reference | Block states, pool kinds, segment lifecycle, and caching allocator internals |
| [PyTorch CUDA Graphs](public/cuda-graphs/index.html) | Reference | Capture lifecycle, private memory pools, graph-safe RNG, conditional nodes, and Inductor CUDA graph trees |
| [CUDA Graphs Deep Dive](public/cuda-graphs-deep-dive/index.html) | Reference | CUDA 13.5 runtime model: stream capture rules, update, conditional and memory nodes, device launch, field gotchas |

## Development

```
npm install
npm run dev      # local dev server
npm run build    # build to docs/
npm run lint     # eslint
```

Built with React + Vite, deployed to GitHub Pages via `docs/`.