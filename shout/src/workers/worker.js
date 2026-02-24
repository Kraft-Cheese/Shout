import * as Comlink from 'comlink';
import { State } from '../stores/state.js';

/**
 * Worker bridge for Whisper ASR inference.
 * Communicates with whisper.workerThread.js via Comlink.
 */

const SAMPLE_RATE = 16000;

let whisper = null;
let workerInitialized = false;

/**
 * Initialize the worker. Call this on app startup.
 */
export async function initWorker() {
  try {
    const worker = new Worker(
      new URL('./whisper.workerThread.js', import.meta.url),
      { type: 'module' }
    );
    whisper = Comlink.wrap(worker);
    workerInitialized = true;

    // Check if WASM is available
    const { available } = await whisper.checkAvailability();
    if (!available) {
      State.modelStatus.value = 'error';
      State.error.value = 'Whisper WASM not found. Add files to public/wasm/';
      return false;
    }

    State.modelStatus.value = 'idle';
    return true;
  } catch (err) {
    State.modelStatus.value = 'error';
    State.error.value = `Worker initialization failed: ${err.message}`;
    return false;
  }
}

/**
 * Load the ASR model for the specified language.
 * @param {string} language - Language code (e.g., 'en', 'fr')
 */
export async function loadModel(language) {
  if (!workerInitialized || !whisper) {
    State.error.value = 'Worker not initialized. Please refresh the page.';
    return;
  }

  State.modelStatus.value = 'loading';
  State.error.value = null;

  try {
    await whisper.load(language);
    State.modelStatus.value = 'ready';
  } catch (err) {
    State.modelStatus.value = 'error';
    State.error.value = err.message;
  }
}

/**
 * Transcribe audio and update application state.
 * @param {Float32Array} audio - Audio samples at 16kHz
 */
export async function transcribe(audio) {
  if (!workerInitialized || !whisper) {
    State.error.value = 'Worker not initialized.';
    return;
  }

  try {
    const start = performance.now();
    const result = await whisper.transcribe(audio);
    const latency = performance.now() - start;

    // Calculate real-time factor (processing time / audio duration)
    const audioDurationMs = (audio.length / SAMPLE_RATE) * 1000;
    const rtf = latency / audioDurationMs;

    // Update state with results
    State.transcript.value = result.text;
    State.confidence.value = result.confidence;
    State.reconstructed.value = false;
    State.metrics.value = { latency, rtf, inference: latency };
    State.error.value = null;
  } catch (err) {
    State.error.value = `Transcription failed: ${err.message}`;
  }
}