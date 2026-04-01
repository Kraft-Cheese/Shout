import { State, isLoading } from '../stores/state.js';


/**
 * Displays transcription output with token-level confidence highlighting.
 *
 * Token colours:
 *   green  (high) p > 0.7 confident
 *   yellow (medium) p > 0.4  uncertain
 *   red    (low) p <= 0.4 low confidence
 */
export function Output() {
  const confidenceMin = State.confidence.value;
  const confidenceAvg = State.confidenceAvg.value;
  const confidenceLevel =
    confidenceMin > 0.7 ? 'high' : confidenceMin > 0.4 ? 'medium' : 'low';

  const tokens = State.tokens.value;
  const hasTokens = tokens && tokens.length > 0;
  const hasTranscript = State.transcript.value && State.transcript.value.trim();

  return (
    <div class="section output-section">
      <h2>Transcription</h2>

      {isLoading.value ? (
        <div class="processing">
          <span class="spinner" />
          {State.loadPhase.value === 'downloading' ? (
            <div class="download-progress">
              <span>
                Downloading model...{' '}
                {State.downloadProgress.value > 0
                  ? `${(State.downloadProgress.value * 100).toFixed(0)}%`
                  : ''}
              </span>
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  style={{ width: `${(State.downloadProgress.value * 100).toFixed(1)}%` }}
                />
              </div>
            </div>
          ) : State.loadPhase.value === 'cached' ? (
            <span>Loading model from cache...</span>
          ) : State.loadPhase.value === 'initializing' ? (
            <span>Initializing model...</span>
          ) : (
            <span>Loading model...</span>
          )}
        </div>
      ) : State.isProcessing.value ? (
        <div class="processing">
          <span class="spinner" />
          <span>Transcribing...</span>
        </div>
      ) : (
        <>
          <div class={`transcription-box ${!hasTranscript ? 'empty' : ''}`}>
            {!hasTranscript
              ? 'Record or upload audio to begin.'
              : hasTokens
              ? tokens.map((token, i) => {
                  const level =
                    token.p > 0.7 ? 'high' : token.p > 0.4 ? 'medium' : 'low';
                  return (
                    <span
                      key={i}
                      class={`token confidence-${level}`}
                      title={`p=${(token.p * 100).toFixed(1)}%  t0=${token.t0 * 10}ms`}
                    >
                      {token.text}
                    </span>
                  );
                })
              : State.transcript.value}
          </div>

          {hasTranscript && (
            <div class="meta-row">
              <span class={`confidence-badge ${confidenceLevel}`}>
                Min {(confidenceMin * 100).toFixed(0)}%
              </span>
              <span class="metric">Avg {(confidenceAvg * 100).toFixed(0)}%</span>
              {State.reconstructed.value && (
                <span class="badge">Reconstructed</span>
              )}
              <span class="metric">
                {State.metrics.value.latency.toFixed(0)} ms
              </span>
              <span class="metric">
                RTF {State.metrics.value.rtf.toFixed(2)}
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
