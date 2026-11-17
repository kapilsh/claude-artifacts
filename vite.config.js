import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BASE = '/claude-artifacts/'

// The dev server's SPA fallback answers `/claude-artifacts/<slug>/` with the
// root index.html, so artifact links open the landing page again. GitHub Pages
// (and `vite preview`) serve the directory's index.html. Do the same in dev.
function publicDirIndex() {
  return {
    name: 'public-dir-index',
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        const [path, query = ''] = req.url.split('?')
        if (path.startsWith(BASE) && path.endsWith('/') && path !== BASE) {
          const rel = path.slice(BASE.length)
          if (existsSync(join(server.config.publicDir, rel, 'index.html'))) {
            req.url = `${path}index.html${query ? `?${query}` : ''}`
          }
        }
        next()
      })
    },
  }
}

// Deployed to GitHub Pages from docs/ on the default branch.
export default defineConfig({
  base: BASE,
  plugins: [publicDirIndex(), react()],
  build: { outDir: 'docs', emptyOutDir: true },
})
