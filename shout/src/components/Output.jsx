import { State, isLoading } from '../stores/state.js';

/**
 * Displays transcription output with confidence indicator.
 */
export function Output() {
  const confidence = State.confidence.value;
  const confidenceLevel =
    confidence > 0.7 ? 'high' : confidence > 0.4 ? 'medium' : 'low';

  const hasTranscript = State.transcript.value && State.transcript.value.trim();

  return (
    <div class="section output-section">
      <h2>Transcription</h2>

      {isLoading.value ? (
        <div class="processing">
          <span class="spinner" />
          <span>Loading model...</span>
        </div>
      ) : State.isProcessing.value ? (
        <div class="processing">
          <span class="spinner" />
          <span>Transcribing...</span>
        </div>
      ) : (
        <>
          <div class={`transcription-box ${!hasTranscript ? 'empty' : ''}`}>
            {hasTranscript
              ? State.transcript.value
              : 'Record or upload audio to begin.'}
          </div>

          {hasTranscript && (
            <div class="meta-row">
              <span class={`confidence-badge ${confidenceLevel}`}>
                {(confidence * 100).toFixed(0)}% confident
              </span>
              {State.reconstructed.value && (
                <span class="badge">Reconstructed</span>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}