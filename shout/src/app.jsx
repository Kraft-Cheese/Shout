import { useEffect } from 'preact/hooks';
import './app.css';
import { State } from './stores/state.js';
import { initWorker, loadModel } from './workers/worker.js';
import { checkCapabilities } from './libs/capabilities.js';
import { LanguageSelect } from './components/LanguageSelect';
import { Recorder } from './components/Recorder';
import { Output } from './components/Output';
import { UploadControl } from './components/UploadControl.jsx';

/**
 * Main application component for Shout - Privacy-preserving ASR.
 */
export function App() {
  // Initialize worker on mount
  useEffect(() => {
    (async () => {
      const { issues } = await checkCapabilities();
      const blocking = issues.find(i => i.includes('WebAssembly'));
      if (blocking) { State.error.value = blocking; return; }
      if (issues.length > 0) console.warn('[shout] capability warnings:', issues);

      const success = await initWorker();
      if (success) loadModel(State.language.value);
    })();
  }, []);

  return (
    <main id="shout-app">
      <header class="app-toolbar">
        <div class="toolbar-left">
          <h1>SHOUT</h1>
        </div>
      </header>

      <div class="language-float">
        <LanguageSelect />
      </div>

      <div class="main-card">
        {/* Error Banner */}
        {State.error.value && (
          <div class="error-banner">
            <strong>Error</strong>
            {State.error.value}
          </div>
        )}

        <Recorder />
        <Output />
      </div>

      <UploadControl />

    </main>
  );
}
