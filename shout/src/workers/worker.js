import * as Comlink from 'comlink';
import { State } from '../stores/state.js';
import { reconstruct, reconstructTokens } from '../lib/reconstruct.js';
import { calculateWER, calculateCER, calculateF1 } from '../lib/metrics.js';

/**
 * Worker bridge for Whisper ASR inference.
 */

const SAMPLE_RATE = 16000;

// Confidence threshold reconstruction fires when segment confidence < TAU.
// Tuned against B1 prediction CSVs; 0.5 is a starting point.
const TAU = 0.5;

// Languages that have a vocab file in public/vocab/ for reconstruction
const RECONSTRUCTABLE = new Set(['sei', 'ncx', 'sei-joint', 'ncx-joint']);

const CONTROL_TAG_RE = /\[_[A-Za-z]+(?:_[0-9]+)?_?\]/g;
const WHISPER_SPECIAL_RE = /<\|[^|]+\|>/g;

let whisper = null;
let workerInitialized = false;

// Vocab cache fetched once per language on model load
const vocabCache = {};

async function fetchVocab(language) {
  const targetLang = (language === 'sei-joint' || language === 'ncx-joint') ? language.split('-')[0] : language;
  if (vocabCache[targetLang]) return vocabCache[targetLang];
  try {
    const res = await fetch(`/vocab/${targetLang}.json`);
    if (!res.ok) return null;
    vocabCache[targetLang] = await res.json();
    return vocabCache[targetLang];
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
    .map((t) => ({ ...t, text: sanitizeTokenText(t.text), reconstructed: t.reconstructed ?? false }))
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
      // console.log('[worker] load progress:', ratio, 'cached:', cached);
      if (cached) {
        // Model is being read from local cache, no download progress needed
        State.loadPhase.value = ratio >= 1.0 ? 'initializing' : 'cached';
        State.downloadProgress.value = null;
      } else if (ratio >= 1.0) {
        State.downloadProgress.value = 1;
        State.loadPhase.value = 'initializing';
      } else {
        State.downloadProgress.value = ratio;
        State.loadPhase.value = 'downloading';
      }
    });

    console.log('[worker] calling whisper.load for', language);
    await whisper.load(language, State.useQuantized.value, onProgress);
    console.log('[worker] whisper.load finished, waiting for vocab');
    await vocabPromise;

    console.log('[worker] load complete, updating state');
    State.loadPhase.value = null;
    State.downloadProgress.value = null;
    State.modelStatus.value = 'ready';
  } catch (err) {
    console.error('[worker] loadModel failed:', err);
    State.loadPhase.value = null;
    State.downloadProgress.value = null;
    State.modelStatus.value = 'error';
    State.error.value = err.message;
  }
}

export async function loadModelFromFile(file) {
  if (!workerInitialized || !whisper) {
    State.error.value = 'Worker not initialized. Please refresh the page.';
    return;
  }

  State.modelStatus.value = 'loading';
  State.loadPhase.value = 'initializing';
  State.downloadProgress.value = null;
  State.error.value = null;

  try {
    const arrayBuffer = await file.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);
    await whisper.loadFromBytes(
      State.language.value,
      State.useQuantized.value,
      Comlink.transfer(bytes, [bytes.buffer])
    );

    State.loadPhase.value = null;
    State.modelStatus.value = 'ready';
  } catch (err) {
    console.error('[worker] loadModelFromFile failed:', err);
    State.loadPhase.value = null;
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
    State.isProcessing.value = true;
    const result = await whisper.transcribe(audio);
    const latency = performance.now() - start;

    const audioDurationMs = (audio.length / SAMPLE_RATE) * 1000;
    const rtf = latency / audioDurationMs;

    const currentLanguage = State.language.value;
    const vocabLang = (currentLanguage === 'sei-joint' || currentLanguage === 'ncx-joint') ? currentLanguage.split('-')[0] : currentLanguage;
    const vocab = vocabCache[vocabLang] ?? null;

    const processResult = (res, applyReconstruction) => {
      const cleanTokens = sanitizeTokens(res.tokens ?? []);
      const tokenText = buildTextFromTokens(cleanTokens);
      const rawText = sanitizeText(res.text);
      const baseText =
        !rawText || (!/\s/.test(rawText) && tokenText)
          ? tokenText
          : rawText;
      const confidenceStats = computeSegmentConfidence(res.confidence, cleanTokens);

      let finalText = baseText;
      let finalTokens = cleanTokens;
      let wasReconstructed = false;

      if (applyReconstruction && confidenceStats.min < TAU && vocab) {
        if (res.tokens && res.tokens.length > 0) {
          const { text, tokens: reconstructedTokens, changed } = reconstructTokens(
            res.tokens,
            vocab,
            vocabLang,
            TAU
          );
          if (changed > 0) {
            finalText = sanitizeText(text);
            finalTokens = sanitizeTokens(reconstructedTokens);
            wasReconstructed = true;
          }
        } else {
          const { text, changed } = reconstruct(baseText, vocab, vocabLang);
          if (changed > 0) {
            finalText = sanitizeText(text);
            wasReconstructed = true;
          }
        }
      }

      return {
        transcript: finalText,
        confidence: confidenceStats.min,
        confidenceAvg: confidenceStats.avg,
        tokens: finalTokens,
        reconstructed: wasReconstructed,
      };
    };

    const mainResult = processResult(result, true);

    if (State.comparisonMode.value) {
      const rawResult = processResult(result, false);

      // For "base english", we need to run transcription with the English model.
      let englishResult = null;
      if (vocabLang === 'en') {
        englishResult = rawResult;
      } else {
        try {
          console.log('[worker] Comparison mode: loading English model');
          await whisper.load('en', State.useQuantized.value);
          console.log('[worker] Comparison mode: transcribing with English model');
          const resEn = await whisper.transcribe(audio);
          englishResult = processResult(resEn, false);
          
          // Restore original model
          console.log('[worker] Comparison mode: restoring original model', currentLanguage);
          await loadModel(currentLanguage);
        } catch (enErr) {
          console.error('[worker] Comparison mode English transcription failed:', enErr);
          englishResult = { 
            transcript: `Error: ${enErr.message}`, 
            confidence: 0, 
            confidenceAvg: 0, 
            tokens: [], 
            reconstructed: false 
          };
          // Try to restore original model even on failure
          await loadModel(currentLanguage).catch(console.error);
        }
      }

      State.comparisonResults.value = {
        withReconstruction: mainResult,
        withoutReconstruction: rawResult,
        baseEnglish: englishResult,
      };
    } else {
      State.comparisonResults.value = null;
    }

    State.transcript.value = mainResult.transcript;
    State.confidence.value = mainResult.confidence;
    State.confidenceAvg.value = mainResult.confidenceAvg;
    State.tokens.value = mainResult.tokens;
    State.reconstructed.value = mainResult.reconstructed;
    State.metrics.value = { latency, rtf, inference: latency };
    State.error.value = null;
  } catch (err) {
    State.error.value = `Transcription failed: ${err.message}`;
  } finally {
    State.isProcessing.value = false;
  }
}

/**
 * Simplified transcription for batch evaluation.
 * Does not update global UI state. Throws on error.
 */
export async function transcribeBatch(audio, targetLang, referenceText = '') {
  if (!workerInitialized || !whisper) {
    throw new Error('Worker not initialized.');
  }

  const start = performance.now();
  const result = await whisper.transcribe(audio);
  const latency = performance.now() - start;

  const audioDurationMs = (audio.length / SAMPLE_RATE) * 1000;
  const rtf = latency / audioDurationMs;

  const vocabLang = (targetLang === 'sei-joint' || targetLang === 'ncx-joint') ? targetLang.split('-')[0] : targetLang;
  const vocab = vocabCache[vocabLang] ?? null;

  const processResult = (res, applyReconstruction) => {
    const cleanTokens = sanitizeTokens(res.tokens ?? []);
    const tokenText = buildTextFromTokens(cleanTokens);
    const rawText = sanitizeText(res.text);
    const baseText = !rawText || (!/\s/.test(rawText) && tokenText) ? tokenText : rawText;
    const confidenceStats = computeSegmentConfidence(res.confidence, cleanTokens);

    let finalText = baseText;
    let wasReconstructed = false;

    if (applyReconstruction && confidenceStats.min < TAU && vocab) {
      if (res.tokens && res.tokens.length > 0) {
        const { text, changed } = reconstructTokens(res.tokens, vocab, vocabLang, TAU);
        if (changed > 0) {
          finalText = sanitizeText(text);
          wasReconstructed = true;
        }
      } else {
        const { text, changed } = reconstruct(baseText, vocab, vocabLang);
        if (changed > 0) {
          finalText = sanitizeText(text);
          wasReconstructed = true;
        }
      }
    }

    return {
      transcript: finalText,
      baseText: baseText,
      confidence: confidenceStats.min,
      reconstructed: wasReconstructed,
    };
  };

  const processed = processResult(result, true);
  const raw = processResult(result, false);

  const calculateAllMetrics = (text) => {
    const m = {
      wer: referenceText ? calculateWER(referenceText, text) : 0,
      cer: referenceText ? calculateCER(referenceText, text) : 0,
    };

    if (referenceText) {
      const f1 = calculateF1(referenceText, text);
      m.f1_morpheme = f1.morphemeF1;
      m.f1_boundary = f1.boundaryF1;
    } else {
      m.f1_morpheme = 0;
      m.f1_boundary = 0;
    }
    return m;
  };

  const rawMetrics = calculateAllMetrics(raw.transcript);
  const finalMetrics = calculateAllMetrics(processed.transcript);

  return {
    rawText: raw.transcript,
    rawMetrics,
    reconstruction: processed.reconstructed,
    finalText: processed.transcript,
    finalMetrics,
    metrics: {
      latency,
      rtf
    }
  };
}
