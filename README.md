# claude-artifacts

Research notes and tools — generated from explorations in AI systems, GPU programming, and distributed training.

Deployed to [kapilsharma.github.io/claude-artifacts](https://kapilsharma.github.io/claude-artifacts/).

## Artifacts

| Artifact | Kind | Description |
|---|---|---|
| [PyTorch Allocator Map](public/pytorch-allocator-map/index.html) | Reference | Block states, pool kinds, segment lifecycle, and caching allocator internals |

## Development

```
npm install
npm run dev      # local dev server
npm run build    # build to docs/
npm run lint     # eslint
```

Built with React + Vite, deployed to GitHub Pages via `docs/`.