import * as Comlink from 'comlink';
import { State } from '../stores/state.js';
import { reconstruct, reconstructTokens } from '../lib/reconstruct.js';

/**
 * Worker bridge for Whisper ASR inference.
 * Communicates with whisper.workerThread.js via Comlink.
 */

const SAMPLE_RATE = 16000;

// Confidence threshold reconstruction fires when segment confidence < TAU.
// Tuned against B1 prediction CSVs; 0.5 is a starting point.
const TAU = 0.5;

// Languages that have a vocab file in public/vocab/ for reconstruction
const RECONSTRUCTABLE = new Set(['sei', 'ncx']);

const CONTROL_TAG_RE = /\[_[A-Za-z]+(?:_[0-9]+)?_?\]/g;
const WHISPER_SPECIAL_RE = /<\|[^|]+\|>/g;

let whisper = null;
let workerInitialized = false;

// Vocab cache fetched once per language on model load
const vocabCache = {};

async function fetchVocab(language) {
  if (vocabCache[language]) return vocabCache[language];
  try {
    const res = await fetch(`/vocab/${language}.json`);
    if (!res.ok) return null;
    vocabCache[language] = await res.json();
    return vocabCache[language];
  } catch {
    return null;
  }
}

function sanitizeText(text) {
  return (text || '')
    .replace(CONTROL_TAG_RE, ' ')
    .replace(WHISPER_SPECIAL_RE, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function sanitizeTokenText(text) {
  return (text || '')
    .replace(CONTROL_TAG_RE, '')
    .replace(WHISPER_SPECIAL_RE, '');
}

function sanitizeTokens(tokens = []) {
  return tokens
    .map((t) => ({ ...t, text: sanitizeTokenText(t.text) }))
    .filter((t) => t.text.trim().length > 0);
}

function buildTextFromTokens(tokens = []) {
  const pieces = tokens
    .map((t) => sanitizeTokenText(t.text))
    .filter((s) => s.trim().length > 0);

  if (pieces.length === 0) return '';

  let out = '';
  for (const piece of pieces) {
    if (!out) {
      out = piece.trimStart();
      continue;
    }

    const prev = out[out.length - 1] || '';
    const next = piece[0] || '';
    const hasBoundarySpace = /\s$/.test(out) || /^\s/.test(piece);
    const needSpace =
      !hasBoundarySpace &&
      /[A-Za-z0-9]/.test(prev) &&
      /[A-Za-z0-9]/.test(next);

    out += needSpace ? ` ${piece}` : piece;
  }

  return out.replace(/\s+/g, ' ').trim();
}

function computeSegmentConfidence(rawConfidence, cleanTokens = []) {
  const probs = cleanTokens
    .filter((t) => /[A-Za-z0-9]/.test(t.text) && Number.isFinite(t.p))
    .map((t) => t.p);

  if (probs.length === 0) {
    return { min: rawConfidence, avg: rawConfidence };
  }

  const min = Math.min(...probs);
  const avg = probs.reduce((sum, p) => sum + p, 0) / probs.length;
  return { min, avg };
}

export async function initWorker() {
  try {
    const worker = new Worker(
      new URL('./whisper.workerThread.js', import.meta.url),
      { type: 'module' }
    );
    whisper = Comlink.wrap(worker);
    workerInitialized = true;

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

export async function loadModel(language) {
  if (!workerInitialized || !whisper) {
    State.error.value = 'Worker not initialized. Please refresh the page.';
    return;
  }

  State.modelStatus.value = 'loading';
  State.loadPhase.value = 'downloading';
  State.downloadProgress.value = 0;
  State.error.value = null;

  try {
    const vocabPromise = RECONSTRUCTABLE.has(language)
      ? fetchVocab(language)
      : Promise.resolve(null);

    const onProgress = Comlink.proxy(({ ratio, cached }) => {
      if (cached) {
        // Model is being read from local cache, no download progress needed
        State.loadPhase.value = ratio >= 1.0 ? 'initializing' : 'cached';
        State.downloadProgress.value = null;
      } else if (ratio >= 1.0) {
        State.downloadProgress.value = 1;
        State.loadPhase.value = 'initializing';
      } else {
        State.downloadProgress.value = ratio;
      }
    });

    await whisper.load(language, State.useQuantized.value, onProgress);
    await vocabPromise;

    State.loadPhase.value = null;
    State.downloadProgress.value = null;
    State.modelStatus.value = 'ready';
  } catch (err) {
    State.loadPhase.value = null;
    State.downloadProgress.value = null;
    State.modelStatus.value = 'error';
    State.error.value = err.message;
  }
}

export async function transcribe(audio) {
  if (!workerInitialized || !whisper) {
    State.error.value = 'Worker not initialized.';
    return;
  }

  try {
    const start = performance.now();
    const result = await whisper.transcribe(audio);
    const latency = performance.now() - start;

    const audioDurationMs = (audio.length / SAMPLE_RATE) * 1000;
    const rtf = latency / audioDurationMs;

    const language = State.language.value;
    const vocab = vocabCache[language] ?? null;
    const cleanTokens = sanitizeTokens(result.tokens ?? []);
    const tokenText = buildTextFromTokens(cleanTokens);
    const rawText = sanitizeText(result.text);
    const baseText =
      !rawText || (!/\s/.test(rawText) && tokenText)
        ? tokenText
        : rawText;
    const confidenceStats = computeSegmentConfidence(result.confidence, cleanTokens);

    let finalText = baseText;
    let wasReconstructed = false;

    // Uncertainty-triggered reconstruction:
    // If segment confidence is below Tau and a vocab exists for this language,
    // apply token-level correction targeting only the uncertain tokens.
    // Falls back to whole-text reconstruction if token data is unavailable.
    if (confidenceStats.min < TAU && vocab) {
      if (result.tokens && result.tokens.length > 0) {
        const { text, changed } = reconstructTokens(
          result.tokens,
          vocab,
          language,
          TAU
        );
        if (changed > 0) {
            finalText = sanitizeText(text);
          wasReconstructed = true;
        }
      } else {
        const { text, changed } = reconstruct(baseText, vocab, language);
        if (changed > 0) {
            finalText = sanitizeText(text);
          wasReconstructed = true;
        }
      }
    }

    State.transcript.value = finalText;
    State.confidence.value = confidenceStats.min;
    State.confidenceAvg.value = confidenceStats.avg;
    State.tokens.value = cleanTokens;
    State.reconstructed.value = wasReconstructed;
    State.metrics.value = { latency, rtf, inference: latency };
    State.error.value = null;
  } catch (err) {
    State.error.value = `Transcription failed: ${err.message}`;
  }
}
