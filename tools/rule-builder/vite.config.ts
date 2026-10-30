import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  // Relative base so the build can be served from any mount point
  // (main.py serves dist under /static/).
  base: './',
  plugins: [vue()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      // `pnpm dev` talks to the backend started by run-rule-builder.sh
      '/api': 'http://localhost:2800',
    },
  },
})
