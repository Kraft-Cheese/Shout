import * as Comlink from 'comlink';

/**
 * Whisper ASR Worker Thread
 *
 * Runs in a separate thread for non-blocking inference.
 * Exposes load() and transcribe() via Comlink.
 *
 * WASM API reference: public/wasm/index.html
 *   - Module.FS_createDataFile()
 *   - Module.init(fname, lang)
 *   - Module.set_audio(ctx, buf)
 *   - Module.get_transcribed()
 *   - Module.get_probabilities()
 *   - Module.get_confidence()
 */

let ctx = null;
let isModelLoaded = false;

/**
 * Check if a file exists by fetching its HEAD.
 */
async function fileExists(url) {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return response.ok;
  } catch {
    return false;
  }
}

const api = {
  /**
   * Check if the WASM module is available.
   */
  async checkAvailability() {
    const wasmExists = await fileExists('/wasm/stream.js');
    return { available: wasmExists };
  },

  /**
   * Load the Whisper model for a given language.
   * @param {string} language - Language code (e.g., 'en')
   */
  async load(language) {
    const wasmUrl = '/wasm/stream.js';
    const wasmExists = await fileExists(wasmUrl);

    if (!wasmExists) {
      throw new Error(
        'Whisper WASM module not found. Please add stream.js to public/wasm/'
      );
    }

    try {
      // Pre-configure the global Module object before importing stream.js.
      // Emscripten checks `typeof Module !== 'undefined'` at the top of the
      // generated file and uses this object if present Ref: stream.js
      // postRun fires once the WASM binary is compiled and ready Ref: index.html
      await new Promise((resolve, reject) => {
        self.Module = {
          print: () => {},
          printErr: () => {},
          setStatus: () => {},
          monitorRunDependencies: () => {},
          postRun: resolve,
          onAbort: () => reject(new Error('WASM aborted during initialisation')),
        };
        import(/* @vite-ignore */ '/wasm/stream.js').catch(reject);
      });

      // Fetch model binary
      const modelUrl = `/models/whisper-base-${language}-q5.bin`;
      const modelExists = await fileExists(modelUrl);

      if (!modelExists) {
        throw new Error(
          `Model not found: ${modelUrl}. Please download the model file.`
        );
      }

      const response = await fetch(modelUrl);
      const buffer = await response.arrayBuffer();

      // Write model binary into the WASM virtual filesystem (MEMFS).
      // Ref: index.html (storeFS function)
      try { self.Module.FS_unlink('whisper.bin'); } catch (_) {}
      self.Module.FS_createDataFile('/', 'whisper.bin', new Uint8Array(buffer), true, true);

      // Initialise whisper with filename + language code.
      // Returns an integer instance ID. Ref: index.html
      ctx = self.Module.init('whisper.bin', language);
      if (!ctx) {
        throw new Error(
          'Module.init returned null'
        );
      }

      isModelLoaded = true;
    } catch (err) {
      isModelLoaded = false;
      throw new Error(`Failed to load Whisper: ${err.message}`);
    }
  },

  /**
   * Check if model is loaded.
   */
  isReady() {
    return isModelLoaded;
  },

  /**
   * Transcribe audio samples.
   * @param {Float32Array} audio - Audio at 16kHz
   * @returns {{ text: string, confidence: number }}
   */
  async transcribe(audio) {
    if (!ctx || !isModelLoaded) {
      throw new Error('Model not loaded. Please load the model first.');
    }

    // Feed audio into the stream context
    self.Module.set_audio(ctx, audio);

    // Poll get_transcribed() until text is returned or 30s timeout.
    // Ref: index.html
    const deadline = Date.now() + 30000;
    let text = null;
    while (Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 100));
      const t = self.Module.get_transcribed();
      if (t && t.length > 0) {
        text = t;
        break;
      }
    }

    let confidence = self.Module.get_confidence();

    if (!text) {
      throw new Error('Transcription timed out.');
    }

    return { text, confidence };
  },
};

Comlink.expose(api);
