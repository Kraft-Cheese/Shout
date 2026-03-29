# Workers

Browser worker runtime for ASR inference using WASM!

## Files

- `worker.js`: Creates the worker thread with Comlink, loads models, runs transcription, sanitizes output text/tokens, computes confidence stats, and applies reconstruction

- `whisper.workerThread.js`: Resolves model URLs, loads model bytes into the Emscripten FS, runs `Module.init`, and executes transcription.

Basically worker does all non-WASM tasks
workerThread handles loading whisper models and running transcription

## Runtime Flow

1. UI calls `initWorker()` in `worker.js`.
2. `worker.js` creates `whisper.workerThread.js` as a module worker and wraps it via Comlink.
3. UI calls `loadModel(language)`.
4. Worker thread initializes WASM (`stream.js`) and loads model bytes into `whisper.bin`.
5. UI sends audio to `transcribe(audio)`.
6. Worker thread returns `{ text, confidence, tokens }`.
7. `worker.js` sanitizes output(removes whisper tokens), computes confidence stats, optionally reconstructs low-confidence segments, then writes to global state.

## Confidence

`worker.js` computes confidence from lexical tokens (word-like tokens with numeric probability):

- `min`: conservative confidence (lowest lexical token probability).
- `avg`: average lexical token probability.

## Notes
`min` is used for reconstruction threshold checks not avg!!!!
This is because we want to lookup regardless


