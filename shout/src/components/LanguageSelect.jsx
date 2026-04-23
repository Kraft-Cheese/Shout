import { State, isReady, isLoading } from '../stores/state.js';
import { loadModel } from '../workers/worker.js';


/** Supported languages for transcription */
const LANGUAGES = [
  { code: 'en',  name: 'English' },
  { code: 'fr',  name: 'French' },
  { code: 'sei', name: 'Seri' },
  { code: 'ncx', name: "Nxa'amxcín" },
  { code: 'sei-joint', name: 'Seri (Joint)' },
  { code: 'ncx-joint', name: "Nxa'amxcín (Joint)" },
];

/**
 * Language selection dropdown.
 * Updates global state and reloads model when changed.
 */
export function LanguageSelect() {
  const handleLangChange = async (e) => {
    State.language.value = e.target.value;
    await loadModel(State.language.value);
  };

  const handleQuantizedChange = async (e) => {
    State.useQuantized.value = e.target.checked;
    await loadModel(State.language.value);
  };

  const handleComparisonChange = (e) => {
    State.comparisonMode.value = e.target.checked;
  };

  return (
    <div class="section language-select">
      <div class="section-label">Language</div>
      <select
        onChange={handleLangChange}
        disabled={isLoading.value}
        value={State.language.value}
      >
        {LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.name}
          </option>
        ))}
      </select>
      
      <div class="controls-row">
        <label class="checkbox-toggle">
          <input
            type="checkbox"
            checked={State.useQuantized.value}
            onChange={handleQuantizedChange}
            disabled={isLoading.value}
          />
          Quantized (Q5)
        </label>

        <label class="checkbox-toggle">
          <input
            type="checkbox"
            checked={State.comparisonMode.value}
            onChange={handleComparisonChange}
          />
          Comparison Mode
        </label>
      </div>
    </div>
  );
}