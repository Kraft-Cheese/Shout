import { useEffect } from 'preact/hooks';
import './app.css';
import { State, isError } from './stores/state.js';
import { initWorker, loadModel } from './workers/worker.js';
import { LanguageSelect } from './components/LanguageSelect';
import { Recorder } from './components/Recorder';
import { Output } from './components/Output';

/**
 * Main application component for Shout - Privacy-preserving ASR.
 */
export function App() {
  // Initialize worker on mount
  useEffect(() => {
    initWorker().then((success) => {
      if (success) {
        // Auto-load default language model
        loadModel(State.language.value);
      }
    });
  }, []);

  return (
    <main id="shout-app">
      <header class="app-header">
        <h1>SHOUT</h1>
        <p>Private, on-device ASR</p>
      </header>

      <div class="main-card">
        {/* Error Banner */}
        {State.error.value && (
          <div class="error-banner">
            <strong>Error</strong>
            {State.error.value}
          </div>
        )}

        <LanguageSelect />
        <Recorder />
        <Output />
      </div>

    </main>
  );
}
