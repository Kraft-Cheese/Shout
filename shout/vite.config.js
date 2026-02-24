import { defineConfig } from 'vite'
import preact from '@preact/preset-vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [preact()],
  server: {
    headers: {
      // Required for whisper.cpp threading
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['whisper.cpp', '@ffmpeg/ffmpeg'],
  },
  build: {
    target: 'esnext',
    chunkSizeWarningLimit: 1000,
  },
})
