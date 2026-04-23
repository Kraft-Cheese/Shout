import { useEffect } from 'preact/hooks';
import './app.css';
import { State, isError } from './stores/state.js';
import { initWorker, loadModel } from './workers/worker.js';
import { checkCapabilities } from './libs/capabilities.js';
import { LanguageSelect } from './components/LanguageSelect';
import { Recorder } from './components/Recorder';
import { Output } from './components/Output';
import { Evaluation } from './components/Evaluation';
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
      if (success) {
        // Auto-load default language model
        loadModel(State.language.value);
      }
    })();
  }, []);

  const navigate = (view) => {
    State.view.value = view;
  };

  return (
    <main id="shout-app">
      <header class="app-toolbar">
        <div class="toolbar-left">
          <h1>SHOUT</h1>
        </div>
        <nav class="app-nav">
          <button 
            class={State.view.value === 'main' ? 'active' : ''} 
            onClick={() => navigate('main')}
          >
            ASR
          </button>
          <button 
            class={State.view.value === 'evaluate' ? 'active' : ''} 
            onClick={() => navigate('evaluate')}
          >
            Evaluate
          </button>
        </nav>
      </header>

      {State.view.value === 'main' && (
        <div class="language-float">
          <LanguageSelect />
        </div>
      )}

      <div class="main-card">
        {/* Error Banner */}
        {State.error.value && (
          <div class="error-banner">
            <strong>Error</strong>
            {State.error.value}
          </div>
        )}

        {State.view.value === 'main' ? (
          <>
            <Recorder />
            <Output />
          </>
        ) : (
          <Evaluation />
        )}
      </div>

      {State.view.value === 'main' && <UploadControl />}

    </main>
  );
}
