import { useRef } from 'preact/hooks';
import { State, isReady, isBusy, isLoading } from '../stores/state.js';
import { useAudioCapture } from '../hooks/useAudio.js';
import { transcribe } from '../workers/worker.js';

const SAMPLE_RATE = 16000;

/**
 * Audio recorder component.
 * Supports microphone recording and file upload.
 */
export function Recorder() {
  const fileInputRef = useRef(null);
  const { startRecording, stopRecording } = useAudioCapture();

  const isDisabled = !isReady.value || isLoading.value;

  /** Toggle recording state */
  const handleRecord = async () => {
    try {
      if (State.isRecording.value) {
        const audio = await stopRecording();
        State.isRecording.value = false;

        if (audio) {
          State.isProcessing.value = true;
          await transcribe(audio);
          State.isProcessing.value = false;
        }
      } else {
        await startRecording();
        State.isRecording.value = true;
      }
    } catch (err) {
      State.error.value = err.message;
      State.isRecording.value = false;
      State.isProcessing.value = false;
    }
  };

  /** Handle uploaded audio file */
  const handleFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      State.isProcessing.value = true;

      const arrayBuffer = await file.arrayBuffer();
      const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const audio = audioBuffer.getChannelData(0);
      await audioCtx.close();

      await transcribe(audio);
    } catch (err) {
      State.error.value = `Failed to process file: ${err.message}`;
    } finally {
      State.isProcessing.value = false;
      e.target.value = '';
    }
  };

  return (
    <div class="section recorder">
      <div class="section-label">Record</div>
      <div class="button-row">
        <button
          class={`btn-record ${State.isRecording.value ? 'recording' : ''}`}
          onClick={handleRecord}
          disabled={isDisabled || State.isProcessing.value}
        >
          {State.isRecording.value ? 'Stop' : 'Record'}
        </button>

        <button
          class="btn-upload"
          onClick={() => fileInputRef.current?.click()}
          disabled={isDisabled || isBusy.value}
        >
          Upload
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          onChange={handleFile}
          hidden
        />
      </div>

      {State.isRecording.value && (
        <div class="recording-indicator">
          <span class="pulse" />
          Recording...
        </div>
      )}
    </div>
  );
}