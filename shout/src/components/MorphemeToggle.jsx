/**
 * Toggle for the Morpheme boundary visualisation
 */

export default function MorphemeToggle({ label, checked = false, onToggle }) {
  const handleToggle = () => {
    onToggle?.(!checked);
  };

  return (
    <div class="morpheme-toggle">
      {label && <span class="morpheme-toggle-label">{label}</span>}
      <button
        type="button"
        onClick={handleToggle}
        role="switch"
        aria-checked={checked}
        aria-label={label || 'Toggle switch'}
        class={`morpheme-switch ${
          checked ? 'is-on' : 'is-off'
        }`}
      >
        <span
          class={`morpheme-switch-thumb ${
            checked ? 'is-on' : 'is-off'
          }`}
        />
      </button>
    </div>
  );
}