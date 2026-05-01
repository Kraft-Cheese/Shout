/**
 *   App shell precached at install,
 *   Vite injects the manifest at build time.
 *
 *   Model binaries (/models/*.bin) NOT precached. Worker
 *   stores them in the 'shout-models-v1' cache via the Cache API after the
 *   first download. This SW serves them cache-first so they survive offline.
 *
 * COOP/COEP headers required for SharedArrayBuffer
 */

import { precacheAndRoute, cleanupOutdatedCaches } from 'workbox-precaching';
import { registerRoute } from 'workbox-routing';
import { CacheFirst, NetworkFirst } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';

// __WB_MANIFEST is replaced by Vite at build time with the hashed asset list.
precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();

// COOP / COEP headers
// Required for SharedArrayBuffer
// This intercepts every navigation and subresource response to inject headers.
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Only add headers for same-origin requests
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    fetch(event.request).then((response) => {
      // Don't modify opaque or error responses
      if (!response || response.status === 0 || response.type === 'opaque') {
        return response;
      }

      const headers = new Headers(response.headers);
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
      headers.set('Cross-Origin-Resource-Policy', 'same-origin');

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    })
  );
});

// Model binaries cache-first, long TTL
// These are written to 'shout-models-v1' by the worker after download.
// Register this route so the SW intercepts requests for cached models and
// serves them without hitting the network.
registerRoute(
  ({ url }) => url.pathname.startsWith('/models/') && url.pathname.endsWith('.bin'),
  new CacheFirst({
    cacheName: 'shout-models-v1',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 10,           // keep up to 10 model files
        maxAgeSeconds: 60 * 60 * 24 * 90, // 90 days
      }),
    ],
  })
);

// WASM runtime cache-first
registerRoute(
  ({ url }) => url.pathname.startsWith('/wasm/'),
  new CacheFirst({
    cacheName: 'shout-wasm-v1',
    plugins: [
      new ExpirationPlugin({ maxEntries: 5, maxAgeSeconds: 60 * 60 * 24 * 30 }),
    ],
  })
);
