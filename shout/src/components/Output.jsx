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

  const renderTranscriptionBox = (text, resTokens) => {
    const localHasTokens = resTokens && resTokens.length > 0;
    const localHasTranscript = text && text.trim();

    return (
      <div class={`transcription-box ${!localHasTranscript ? 'empty' : ''}`}>
        {!localHasTranscript
          ? 'No result available.'
          : localHasTokens
          ? resTokens.map((token, i) => {
              const level =
                token.p > 0.7 ? 'high' : token.p > 0.4 ? 'medium' : 'low';
              return (
                <span
                  key={i}
                  class={`token confidence-${level}`}
                  title={`p=${(token.p * 100).toFixed(1)}%`}
                >
                  {token.text}
                </span>
              );
            })
          : text}
      </div>
    );
  };

  const renderMetaRow = (res) => {
    const level = res.confidence > 0.7 ? 'high' : res.confidence > 0.4 ? 'medium' : 'low';
    return (
      <div class="meta-row">
        <span class={`confidence-badge ${level}`}>
          Min {(res.confidence * 100).toFixed(0)}%
        </span>
        <span class="metric">Avg {(res.confidenceAvg * 100).toFixed(0)}%</span>
        {res.reconstructed && (
          <span class="badge">Reconstructed</span>
        )}
      </div>
    );
  };

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
      ) : State.comparisonMode.value && State.comparisonResults.value ? (
        <div class="comparison-container">
          <div class="comparison-item">
            <div class="comparison-label">Selected Language (Reconstructed)</div>
            {renderTranscriptionBox(
              State.comparisonResults.value.withReconstruction.transcript,
              State.comparisonResults.value.withReconstruction.tokens
            )}
            {renderMetaRow(State.comparisonResults.value.withReconstruction)}
          </div>

          <div class="comparison-item">
            <div class="comparison-label">Selected Language (Raw)</div>
            {renderTranscriptionBox(
              State.comparisonResults.value.withoutReconstruction.transcript,
              State.comparisonResults.value.withoutReconstruction.tokens
            )}
            {renderMetaRow(State.comparisonResults.value.withoutReconstruction)}
          </div>

          <div class="comparison-item">
            <div class="comparison-label">Base English</div>
            {renderTranscriptionBox(
              State.comparisonResults.value.baseEnglish.transcript,
              State.comparisonResults.value.baseEnglish.tokens
            )}
            {renderMetaRow(State.comparisonResults.value.baseEnglish)}
          </div>
          
          <div class="meta-row" style={{ marginTop: '0.5rem', borderTop: '1px solid var(--her-sand)', paddingTop: '0.75rem' }}>
             <span class="metric">
                {State.metrics.value.latency.toFixed(0)} ms
              </span>
              <span class="metric">
                RTF {State.metrics.value.rtf.toFixed(2)}
              </span>
          </div>
        </div>
      ) : (
        <>
          {renderTranscriptionBox(State.transcript.value, tokens)}

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
