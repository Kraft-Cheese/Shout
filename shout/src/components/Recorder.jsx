import { State, isReady, isLoading } from '../stores/state.js';
import { useAudioCapture } from '../hooks/useAudio.js';
import { transcribe } from '../workers/worker.js';
import { Microphone } from '@phosphor-icons/react';

/**
 * Audio recorder component.
 * Supports microphone recording.
 */
export function Recorder() {
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

  return (
    <div class="section recorder">
      <button
        type="button"
        class={`record-fab ${State.isRecording.value ? 'recording' : ''}`}
        onClick={handleRecord}
        disabled={isDisabled || State.isProcessing.value}
        aria-label={State.isRecording.value ? 'Stop recording' : 'Start recording'}
        title={State.isRecording.value ? 'Stop recording' : 'Start recording'}
      >
        <Microphone size={24} weight="fill" class="record-fab-icon" aria-hidden="true" />
      </button>
    </div>
  );
}