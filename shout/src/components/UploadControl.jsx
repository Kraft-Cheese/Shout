import { useEffect, useRef, useState } from 'preact/hooks';
import { UploadSimple } from '@phosphor-icons/react';
import { State, isBusy, isLoading } from '../stores/state.js';
import { getModelUploadLimit } from '../libs/capabilities.js';
import { loadModelFromFile, transcribe } from '../workers/worker.js';

const SAMPLE_RATE = 16000;
const SUPPORTED_MODEL_EXTS = ['.bin', '.gguf', '.ggml'];

function formatMb(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

function isAudioFile(file) {
  if (!file) return false;
  if (file.type?.startsWith('audio/')) return true;
  return /\.(wav|mp3|m4a|ogg|flac|aac|webm)$/i.test(file.name || '');
}

function getExtension(filename = '') {
  const i = filename.lastIndexOf('.');
  if (i < 0) return '';
  return filename.slice(i).toLowerCase();
}

function startsWithAscii(bytes, text) {
  if (!bytes || bytes.length < text.length) return false;
  for (let i = 0; i < text.length; i += 1) {
    if (bytes[i] !== text.charCodeAt(i)) return false;
  }
  return true;
}

async function readHeader(file, count = 16) {
  const chunk = file.slice(0, count);
  const ab = await chunk.arrayBuffer();
  return new Uint8Array(ab);
}

async function validateModelFile(file) {
  const ext = getExtension(file.name);
  if (!SUPPORTED_MODEL_EXTS.includes(ext)) {
    return {
      ok: false,
      reason: `Unsupported model file extension "${ext || 'none'}". Supported model files: .bin, .gguf, .ggml.`,
    };
  }

  const header = await readHeader(file, 8);

  if (startsWithAscii(header, '<!DO') || startsWithAscii(header, '<html')) {
    return {
      ok: false,
      reason: 'Unsupported model file: upload appears to be HTML, not a Whisper model binary.',
    };
  }

  if (ext === '.gguf' && !startsWithAscii(header, 'GGUF')) {
    return {
      ok: false,
      reason: 'Unsupported model file: .gguf files must start with GGUF header bytes.',
    };
  }

  if (ext === '.ggml' && !startsWithAscii(header, 'ggml') && !startsWithAscii(header, 'GGML')) {
    return {
      ok: false,
      reason: 'Unsupported model file: .ggml files must start with ggml header bytes.',
    };
  }

  return { ok: true };
}

export function UploadControl() {
  const inputRef = useRef(null);
  const [maxModelBytes, setMaxModelBytes] = useState(180 * 1024 * 1024);

  useEffect(() => {
    (async () => {
      try {
        const { maxBytes } = await getModelUploadLimit();
        setMaxModelBytes(maxBytes);
      } catch {
        // Keep conservative default if estimate API is unavailable.
      }
    })();
  }, []);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      State.error.value = null;

      if (isAudioFile(file)) {
        if (State.modelStatus.value !== 'ready') {
          throw new Error('Load a model before uploading audio.');
        }

        State.isProcessing.value = true;
        const arrayBuffer = await file.arrayBuffer();
        const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const audio = audioBuffer.getChannelData(0);
        await audioCtx.close();

        await transcribe(audio);
      } else {
        const validation = await validateModelFile(file);
        if (!validation.ok) {
          throw new Error(validation.reason);
        }

        if (file.size > maxModelBytes) {
          throw new Error(
            `Model file too large (${formatMb(file.size)}). Device limit is ${formatMb(maxModelBytes)}.`
          );
        }
        await loadModelFromFile(file);
      }
    } catch (err) {
      State.error.value = err.message;
    } finally {
      State.isProcessing.value = false;
      e.target.value = '';
    }
  };

  return (
    <div class="upload-control" title={`Upload audio or model. Max model: ${formatMb(maxModelBytes)}`}>
      <button
        type="button"
        class="upload-icon-btn"
        onClick={() => inputRef.current?.click()}
        disabled={isLoading.value || isBusy.value}
        aria-label="Upload audio or model"
      >
        <UploadSimple size={18} weight="bold" aria-hidden="true" />
      </button>
      <span class="upload-limit">Model max {formatMb(maxModelBytes)}</span>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.bin,.gguf,.ggml"
        onChange={handleUpload}
        hidden
      />
    </div>
  );
}
