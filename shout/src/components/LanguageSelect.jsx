import { State, isReady, isLoading } from '../stores/state.js';
import { loadModel } from '../workers/worker.js';

/** Supported languages for transcription */
const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'fr', name: 'French' },
  { code: 'nahuatl', name: 'Nahuatl' },
];

/**
 * Language selection dropdown.
 * Updates global state and reloads model when changed.
 */
export function LanguageSelect() {
  const handleChange = async (e) => {
    const newLang = e.target.value;
    State.language.value = newLang;
    await loadModel(newLang);
  };

  return (
    <div class="section language-select">
      <div class="section-label">Language</div>
      <select
        onChange={handleChange}
        disabled={isLoading.value}
        value={State.language.value}
      >
        {LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.name}
          </option>
        ))}
      </select>
    </div>
  );
}