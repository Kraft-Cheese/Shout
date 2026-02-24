import * as Comlink from 'comlink';

/**
 * Whisper ASR Worker Thread
 *
 * Runs in a separate thread for non-blocking inference.
 * Exposes load() and transcribe() via Comlink.
 */

let ctx = null;
let Module = null;
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
    const wasmExists = await fileExists('/wasm/whisper.js');
    return { available: wasmExists };
  },

  /**
   * Load the Whisper model for a given language.
   * @param {string} language - Language code (e.g., 'en')
   */
  async load(language) {
    // Check if WASM exists first
    const wasmUrl = '/wasm/whisper.js';
    const wasmExists = await fileExists(wasmUrl);

    if (!wasmExists) {
      throw new Error(
        'Whisper WASM module not found. Please add whisper.js to public/wasm/'
      );
    }

    try {
      // Dynamic import with variable to bypass Vite's static analysis
      const wasmPath = wasmUrl;
      Module = await import(/* @vite-ignore */ wasmPath);
      await Module.default();

      // Fetch and initialize model
      const modelUrl = `/models/whisper-base-${language}-q5.bin`;
      const modelExists = await fileExists(modelUrl);

      if (!modelExists) {
        throw new Error(
          `Model not found: ${modelUrl}. Please download the model file.`
        );
      }

      const response = await fetch(modelUrl);
      const buffer = await response.arrayBuffer();
      ctx = Module.init(new Uint8Array(buffer));
      isModelLoaded = true;
    } catch (err) {
      isModelLoaded = false;
      throw new Error(
        `Failed to load Whisper: ${err.message}`
      );
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

    const result = self.Module.transcribe(ctx, audio);

    // Calculate confidence from token probabilities
    const confidence = result.tokens?.length
      ? Math.exp(
          result.tokens.reduce((sum, t) => sum + (t.p || 0), 0) /
          result.tokens.length
        )
      : 0.5;

    return { text: result.text, confidence };
  },
};

Comlink.expose(api);
