import { signal, computed } from '@preact/signals';

/**
 * Global state management for the Shout ASR application.
 * Uses Preact signals for reactive state updates.
 */

// Core application state
export const State = {
  /** Selected transcription language */
  language: signal('en'),

  /** Model loading status: 'idle' | 'loading' | 'ready' | 'error' */
  modelStatus: signal('idle'),

  /** Recording state flags */
  isRecording: signal(false),
  isProcessing: signal(false),

  /** Transcription results */
  transcript: signal(''),
  confidence: signal(0.0),
  reconstructed: signal(false),

  /** Performance metrics */
  metrics: signal({
    inference: 0,
    rtf: 0,
    latency: 0,
  }),

  /** Error message for UI display */
  error: signal(null),
};

// Derived state (computed signals)
export const isReady = computed(() => State.modelStatus.value === 'ready');
export const isLoading = computed(() => State.modelStatus.value === 'loading');
export const isIdle = computed(() => State.modelStatus.value === 'idle');
export const isError = computed(() => State.modelStatus.value === 'error');
export const isBusy = computed(() => State.isRecording.value || State.isProcessing.value);
export const canRecord = computed(() =>
  State.modelStatus.value === 'ready' &&
  !State.isRecording.value &&
  !State.isProcessing.value
);