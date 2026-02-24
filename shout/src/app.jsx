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
        <h1>Shout</h1>
        <p>Privacy-preserving ASR</p>
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

      <footer class="app-footer">
        <small>
          All processing happens locally in your browser.
          No data is sent to any server.
        </small>
      </footer>
    </main>
  );
}
