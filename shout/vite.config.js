import { defineConfig } from 'vite'
import preact from '@preact/preset-vite'
import { VitePWA } from 'vite-plugin-pwa'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    preact(),
    VitePWA({
      // Use injectManifest so we control the service worker fully.
      // The SW file is src/sw.js; Vite compiles + injects the precache manifest.
      strategies: 'injectManifest',
      srcDir: 'src',
      filename: 'sw.js',
      registerType: 'autoUpdate',
      injectManifest: {
        // Cache the app shell but NOT model binaries (too large for precache).
        // Models are handled by the worker via the Cache API on first use.
        globPatterns: ['**/*.{js,css,html,svg,json,wasm}'],
        globIgnores: ['**/models/**'],
        maximumFileSizeToCacheInBytes: 10 * 1024 * 1024, // 10 MB (covers stream.js)
      },
      manifest: {
        name: 'Shout — On-Device ASR',
        short_name: 'Shout',
        description: 'Private, offline speech recognition for low-resource languages',
        theme_color: '#d95030',
        background_color: '#faf7f2',
        display: 'standalone',
        orientation: 'portrait',
        start_url: '/',
        icons: [
          { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
    }),
  ],
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
