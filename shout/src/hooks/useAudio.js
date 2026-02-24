import { useRef, useEffect } from 'preact/hooks';

const SAMPLE_RATE = 16000;

/**
 * Hook for capturing audio from the microphone.
 *
 * @returns {{ startRecording: () => Promise, stopRecording: () => Promise<Float32Array|null> }}
 */
export function useAudioCapture() {
  const chunksRef = useRef([]);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => cleanup();
  }, []);

  function cleanup() {
    if (mediaRecorderRef.current?.state !== 'inactive') {
      try {
        mediaRecorderRef.current?.stop();
      } catch {
        // Ignore errors during cleanup
      }
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    mediaRecorderRef.current = null;
    chunksRef.current = [];
  }

  /**
   * Start recording audio from the microphone.
   */
  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Microphone access not supported in this browser');
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;

    const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    mediaRecorderRef.current = mediaRecorder;

    mediaRecorder.ondataavailable = (e) => {
      if (e.data?.size > 0) {
        chunksRef.current.push(e.data);
      }
    };

    mediaRecorder.start();
  }

  /**
   * Stop recording and return audio as Float32Array at 16kHz.
   * @returns {Promise<Float32Array|null>}
   */
  async function stopRecording() {
    const mediaRecorder = mediaRecorderRef.current;
    if (!mediaRecorder) return null;

    // Wait for recorder to stop
    await new Promise((resolve) => {
      if (mediaRecorder.state === 'inactive') {
        resolve();
        return;
      }
      mediaRecorder.addEventListener('stop', resolve, { once: true });
      try {
        mediaRecorder.stop();
      } catch {
        resolve();
      }
    });

    // Convert chunks to ArrayBuffer
    const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
    const arrayBuffer = await blob.arrayBuffer();

    // Reset state
    chunksRef.current = [];
    mediaRecorderRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    // Decode to 16kHz Float32Array for Whisper
    const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const audio = audioBuffer.getChannelData(0);
    await audioCtx.close();

    return audio;
  }

  return { startRecording, stopRecording };
}