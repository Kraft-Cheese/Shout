import { State, isLoading } from '../stores/state.js';
import { buildInterpunctParts } from '../libs/morpheme.js';
import { DotIcon } from '@phosphor-icons/react';
import MorphemeToggle from './MorphemeToggle.jsx';

// p > 0.7 → high, p > 0.4 → medium, p ≤ 0.4 → low
const confidenceLevel = (p) => p > 0.7 ? 'high' : p > 0.4 ? 'medium' : 'low';

export function Output() {
  const confidenceMin = State.confidence.value;
  const confidenceAvg = State.confidenceAvg.value;
  const showInterpunct = State.showInterpunct.value;

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
              ? (showInterpunct ? buildInterpunctParts(tokens) : tokens).map((item, i) =>
                  item.kind === 'separator' ? (
                    <span key={item.key ?? i} class={`interpunct-dot ${item.className || ''}`}>
                      <DotIcon size={20} weight="regular" aria-hidden="true" />
                    </span>
                  ) : (
                    <span
                      key={item.key ?? i}
                      class={item.className ?? `token confidence-${confidenceLevel(item.p)}`}
                      title={item.title ?? `p=${(item.p * 100).toFixed(1)}%  t0=${item.t0 * 10}ms`}
                    >
                      {item.text}
                    </span>
                  )
                )
              : State.transcript.value}
          </div>

          {hasTranscript && (
            <>
              <div class="toggle-row">
                <MorphemeToggle
                  label="Interpunct"
                  checked={showInterpunct}
                  onToggle={(state) => (State.showInterpunct.value = state)}
                />
              </div>
              <div class="meta-row">
                <span class={`confidence-badge ${confidenceLevel(confidenceMin)}`}>
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
            </>
          )}
        </>
      )}
    </div>
  );
}
