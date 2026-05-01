import * as Comlink from 'comlink';

/**
 * Whisper ASR Worker Thread
 *
 * Runs in a separate thread for non-blocking inference.
 * Exposes load() and transcribe() via Comlink.
 *
 * WASM API reference: public/wasm/index.html
 * - Module.FS_createDataFile()
 * - Module.init(fname, lang)
 * - Module.set_audio(ctx, buf)
 * - Module.get_transcribed()     consumes and returns transcribed text
 * - Module.get_confidence()      segment-level min-token-p, cube-rooted
 * - Module.get_tokens()          [{text, p, t0, t1}] per-token data
 * - Module.set_max_tokens(n)     max output tokens per segment
 * - Module.set_vad_thold(f)      VAD speech probability threshold
 */

// Languages that benefit from higher max_tokens due to polysynthetic morphology
const POLYSYNTHETIC_LANGS = new Set(['sei', 'ncx', 'nhn', 'cr', 'iku', 'sei-joint', 'ncx-joint']);

// Mobile detection to pick a smaller model
const IS_MOBILE = /Mobi|Android/i.test(
  typeof navigator !== 'undefined' ? navigator.userAgent : ''
);

let ctx = null;
let isModelLoaded = false;
let wasmReady = false;

function initFromBuffer(language, buffer) {
  console.log('[whisper] Writing model to WASM FS...');
  try { fsUnlinkSafe('whisper.bin'); } catch (_) {}
  fsCreateDataFileSafe('/', 'whisper.bin', buffer, true, true);

  const modelFsSize = fsStatSizeSafe('/whisper.bin');
  console.log('[whisper] Model in FS size=', modelFsSize ?? 'unknown');

  console.log('[whisper] Calling Module.init...');
  
  // Ported from UI_improvement: Free existing context to prevent memory leaks
  if (ctx && typeof self.Module.free === 'function') {
    try {
      self.Module.free(ctx);
      console.log('[whisper] Freed previous context');
    } catch (e) {
      console.warn('[whisper] Module.free(ctx) failed:', e);
    }
  }
  ctx = null;
  
  let initLanguageUsed = null;
  let initModelPathUsed = null;
  const initLanguages = getInitLanguageFallbacks(language);
  const initModelPaths = ['whisper.bin', '/whisper.bin'];

  for (const initPath of initModelPaths) {
    for (const initLang of initLanguages) {
      const candidateCtx = self.Module.init(initPath, initLang);
      console.log('[whisper] Module.init attempt path=', initPath, 'lang=', initLang, 'ctx=', candidateCtx);
      if (candidateCtx) {
        ctx = candidateCtx;
        initLanguageUsed = initLang;
        initModelPathUsed = initPath;
        break;
      }
    }
    if (ctx) break;
  }

  if (!ctx) {
    throw new Error(
      `Module.init returned null for all attempts: paths=[${initModelPaths.join(', ')}], languages=[${initLanguages.join(', ')}]`
    );
  }

  if (initModelPathUsed !== 'whisper.bin') {
    console.warn(
      '[whisper] Relative model path failed at init; using fallback path',
      initModelPathUsed
    );
  }

  if (initLanguageUsed !== language) {
    console.warn(
      '[whisper] Requested language',
      language,
      'failed at init; using fallback',
      initLanguageUsed
    );
  }

  if (POLYSYNTHETIC_LANGS.has(language)) {
    console.log('[whisper] Polysynthetic language, setting max_tokens=128');
    self.Module.set_max_tokens(128);
  }
}

function getInitLanguageFallbacks(language) {
  const queue = [language];
  if (language !== 'auto') queue.push('auto');
  if (language !== 'en') queue.push('en');
  return [...new Set(queue)];
}

function fsUnlinkSafe(path) {
  if (typeof self.Module?.FS_unlink === 'function') {
    self.Module.FS_unlink(path);
    return;
  }
  if (typeof self.Module?.FS?.unlink === 'function') {
    self.Module.FS.unlink(path);
  }
}

function fsCreateDataFileSafe(dir, name, data, canRead, canWrite) {
  if (typeof self.Module?.FS_createDataFile === 'function') {
    self.Module.FS_createDataFile(dir, name, data, canRead, canWrite);
    return;
  }
  if (typeof self.Module?.FS?.createDataFile === 'function') {
    self.Module.FS.createDataFile(dir, name, data, canRead, canWrite);
    return;
  }
  throw new Error('No Emscripten FS createDataFile API found on Module');
}

function fsStatSizeSafe(path) {
  if (typeof self.Module?.FS_stat === 'function') {
    return self.Module.FS_stat(path)?.size ?? null;
  }
  if (typeof self.Module?.FS?.stat === 'function') {
    return self.Module.FS.stat(path)?.size ?? null;
  }
  return null;
}

async function headInfo(url) {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    const contentLength = response.headers.get('Content-Length');
    const contentType = response.headers.get('Content-Type') || '';
    return {
      ok: response.ok,
      status: response.status,
      length: contentLength ? parseInt(contentLength, 10) : null,
      contentType,
    };
  } catch (err) {
    return {
      ok: false,
      status: 0,
      length: null,
      contentType: '',
      error: err.message,
    };
  }
}

function getModelUrlCandidates(modelSize, language, quantized) {
  const q = quantized ? '-q5' : '';
  const actualLang = (language === 'sei-joint' || language === 'ncx-joint') ? 'joint' : language;
  console.log('[whisper] getModelUrlCandidates modelSize=', modelSize, 'language=', language, 'quantized=', quantized, 'actualLang=', actualLang);
  const candidates = [
    `/models/whisper-${modelSize}-${actualLang}${q}.bin`,
    `/models/whisper-${modelSize}.${actualLang}${q}.bin`,
  ];

  if (language !== 'en') {
    candidates.push(`/models/whisper-${modelSize}${q}.bin`);
    // English-base fallbacks — the repo ships whisper-<size>.en-q5.bin
    candidates.push(`/models/whisper-${modelSize}.en${q}.bin`);
    candidates.push(`/models/whisper-${modelSize}-en${q}.bin`);
  }
  return [...new Set(candidates)];
}

async function resolveModelUrl(modelSize, language, quantized) {
  let candidates = getModelUrlCandidates(modelSize, language, quantized);

  const tryResolve = async (urls) => {
    for (const url of urls) {
      const info = await headInfo(url);
      console.log('[whisper] Model HEAD', url, 'status=', info.status, 'len=', info.length, 'type=', info.contentType || 'unknown');

      const isLikelyBinary =
        info.ok &&
        !info.contentType.includes('text/html') &&
        (info.length === null || info.length > 1024 * 1024);

      if (isLikelyBinary) {
        return url;
      }
    }
    return null;
  };

  let resolved = await tryResolve(candidates);
  if (resolved) return resolved;

  throw new Error(
    `No model binary found. Tried: ${candidates.join(', ')}`
  );
}

// Load the WASM runtime once. Subsequent calls are no-ops because the ES
// module is cached and onRuntimeInitialized won't fire again; we track
// readiness with wasmReady so we don't hang on a never-resolving promise.
async function ensureWasmLoaded() {
  console.log('[whisper] ensureWasmLoaded called, wasmReady=', wasmReady);
  if (wasmReady) {
    console.log('[whisper] WASM already ready, skipping init');
    return;
  }

  const wasmInfo = await headInfo('/wasm/stream.js');
  if (!wasmInfo.ok) {
    throw new Error(
      'Whisper WASM module not found. Please add stream.js to public/wasm/'
    );
  }

  console.log('[whisper] Starting WASM module init...');

  let watchdog = null;

  await new Promise((resolve, reject) => {
    const streamScriptUrl = new URL('/wasm/stream.js', self.location.href).href;

    self.Module = {
      print: (msg) => console.log('[wasm stdout]', msg),
      printErr: (msg) => console.warn('[wasm stderr]', msg),
      setStatus: (msg) => { if (msg) console.log('[wasm setStatus]', msg); },
      monitorRunDependencies: (n) => console.log('[wasm deps remaining]', n),
      // Force pthread workers to load stream.js rather than this wrapper worker.
      mainScriptUrlOrBlob: streamScriptUrl,
      onRuntimeInitialized: () => {
        if (wasmReady) return; // Already handled
        clearTimeout(watchdog);
        console.log('[whisper] onRuntimeInitialized fired WASM heap ready');
        wasmReady = true;
        resolve();
      },
      onAbort: (reason) => {
        clearTimeout(watchdog);
        console.error('[whisper] WASM aborted, reason:', reason);
        reject(new Error('WASM aborted during initialisation'));
      },
    };

    watchdog = setTimeout(() => {
      console.error('[whisper] WATCHDOG: onRuntimeInitialized did not fire after 15s');
      reject(new Error('WASM init timed out onRuntimeInitialized never fired'));
    }, 15000);

    console.log('[whisper] Fetching stream.js for indirect eval from', streamScriptUrl);
    fetch(streamScriptUrl)
      .then((r) => r.text())
      .then((code) => {
        console.log('[whisper] Running stream.js via indirect eval');
        // eslint-disable-next-line no-eval
        (0, eval)(code);
      })
      .catch((err) => {
        clearTimeout(watchdog);
        console.error('[whisper] stream.js fetch/eval failed:', err);
        reject(err);
      });
  });

  console.log('[whisper] ensureWasmLoaded done');
}

const api = {
  async checkAvailability() {
    const info = await headInfo('/wasm/stream.js');
    return { available: info.ok };
  },

  async load(language, quantized = false, onProgress = null) {
    console.log('[whisper] load() called, language=', language, 'quantized=', quantized);
    const actualLang = (language === 'sei-joint' || language === 'ncx-joint') ? 'joint' : language;
    try {
      await ensureWasmLoaded();
      console.log('[whisper] WASM ready, proceeding to model fetch');

      // Fine-tuned LoRA models are whisper-small; generic languages fall back
      // to whisper-tiny (mobile) or whisper-base (desktop).
      const FINE_TUNED_LANGS = new Set(['sei', 'ncx', 'sei-joint', 'ncx-joint', 'joint']);
      const modelSize = FINE_TUNED_LANGS.has(actualLang)
        ? 'small'
        : IS_MOBILE ? 'tiny' : 'base';
      const modelUrl = await resolveModelUrl(modelSize, language, quantized);
      console.log('[whisper] Model URL resolved to:', modelUrl, 'IS_MOBILE=', IS_MOBILE);

      const MODEL_CACHE = 'shout-models-v1';

      // Check if the model is already in the Cache API (persists across sessions).
      let buffer = null;
      if (typeof caches !== 'undefined') {
        try {
          const cache = await caches.open(MODEL_CACHE);
          const cached = await cache.match(modelUrl);
          if (cached) {
            console.log('[whisper] Model found in cache, loading from buffer...');
            if (onProgress) onProgress({ ratio: 0.99, cached: true });
            const ab = await cached.arrayBuffer();
            buffer = new Uint8Array(ab);
            console.log('[whisper] Model loaded from cache, bytes=', buffer.byteLength);
            if (onProgress) onProgress({ ratio: 1.0, cached: true });
          }
        } catch (cacheErr) {
          console.warn('[whisper] Cache read failed, will fetch from network:', cacheErr.message);
        }
      }

      if (!buffer) {
        console.log('[whisper] Fetching model from network:', modelUrl);
        const response = await fetch(modelUrl);
        if (!response.ok) {
          throw new Error(`Model fetch failed for ${modelUrl}: HTTP ${response.status}`);
        }

        const responseType = response.headers.get('Content-Type') || '';
        if (responseType.includes('text/html')) {
          throw new Error(
            `Model fetch returned HTML for ${modelUrl}. Check filename/path and dev-server static file routing.`
          );
        }

        const contentLength = response.headers.get('Content-Length');
        const total = contentLength ? parseInt(contentLength, 10) : null;
        console.log('[whisper] Content-Length:', total);

        const reader = response.body.getReader();
        const chunks = [];
        let received = 0;
        let lastReportedPct = -1;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          received += value.byteLength;
          if (onProgress && total) {
            const pct = Math.floor((received / total) * 100);
            if (pct > lastReportedPct) {
              lastReportedPct = pct;
              console.log('[whisper] Download progress:', pct, '%');
              onProgress({ received, total, ratio: received / total, cached: false });
            }
          }
        }

        console.log('[whisper] Download complete, received', received, 'bytes');
        if (received < 1024 * 1024) {
          throw new Error(
            `Downloaded model is too small (${received} bytes). This usually means an HTML fallback or wrong model filename.`
          );
        }

        if (onProgress) onProgress({ received, total: received, ratio: 1.0, cached: false });

        buffer = new Uint8Array(received);
        let offset = 0;
        for (const chunk of chunks) {
          buffer.set(chunk, offset);
          offset += chunk.byteLength;
        }

        // Store in Cache API for offline use on subsequent loads.
        if (typeof caches !== 'undefined') {
          try {
            const cache = await caches.open(MODEL_CACHE);
            await cache.put(
              modelUrl,
              new Response(buffer.buffer, {
                headers: { 'Content-Type': 'application/octet-stream' },
              })
            );
            console.log('[whisper] Model stored in cache for offline use.');
          } catch (cacheErr) {
            console.warn('[whisper] Cache write failed (non-fatal):', cacheErr.message);
          }
        }
      }

      initFromBuffer(language, buffer);

      isModelLoaded = true;
      console.log('[whisper] load() complete model ready');
    } catch (err) {
      console.error('[whisper] load() failed:', err);
      isModelLoaded = false;
      throw err;
    }
  },

  async loadFromBytes(language, _quantized = false, bytes) {
    console.log('[whisper] loadFromBytes() called, language=', language, 'bytes=', bytes?.byteLength ?? 0);
    try {
      await ensureWasmLoaded();
      const buffer = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
      if (buffer.byteLength < 1024 * 1024) {
        throw new Error(`Uploaded model is too small (${buffer.byteLength} bytes).`);
      }

      initFromBuffer(language, buffer);
      
      isModelLoaded = true;
      console.log('[whisper] loadFromBytes() complete model ready');
    } catch (err) {
      console.error('[whisper] loadFromBytes() failed:', err);
      isModelLoaded = false;
      throw new Error(`Failed to load uploaded model: ${err.message}`);
    }
  },

  isReady() {
    return isModelLoaded;
  },

  async transcribe(audio, language = null) {
    console.log('[whisper] transcribe() called, audio samples=', audio.length, 'language=', language);
    if (!ctx || !isModelLoaded) {
      throw new Error('Model not loaded. Please load the model first.');
    }

    if (language) {
      // Logic for re-initializing language mid-session omitted as per original file
    }

    self.Module.set_audio(ctx, audio);

    const deadline = Date.now() + 30000;
    let text = null;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 100));
      const t = self.Module.get_transcribed();
      if (t && t.length > 0) {
        text = t;
        break;
      }
    }

    if (!text) {
      throw new Error('Transcription timed out.');
    }

    const confidence = self.Module.get_confidence();
    const tokens = self.Module.get_tokens();
    console.log('[whisper] transcribe() result: text=', text, 'confidence=', confidence, 'tokens=', tokens?.length);

    return { text, confidence, tokens };
  },
};

Comlink.expose(api);