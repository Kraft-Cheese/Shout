/**
 * Whisper model sources and URLs.
 */

const HF_BASE = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main';

/** Model configurations with download URLs */
export const MODEL_SOURCES = {
  tiny: {
    name: 'Tiny',
    size: '31 MB',
    url: `${HF_BASE}/ggml-tiny-q5_1.bin`,
  },
  base: {
    name: 'Base',
    size: '57 MB',
    url: `${HF_BASE}/ggml-base-q5_1.bin`,
  },
  small: {
    name: 'Small',
    size: '181 MB',
    url: `${HF_BASE}/ggml-small-q5_1.bin`,
  },
};

/**
 * Get model URL by size.
 * @param {'tiny' | 'base' | 'small'} size
 */
export function getModelUrl(size) {
  return MODEL_SOURCES[size]?.url;
}