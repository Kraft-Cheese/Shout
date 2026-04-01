# Shout Web App

## 1) Build whisper.cpp for WebAssembly

From the whisper.cpp repository root:

1. Install Emscripten and activate it in your shell.
2. Configure and build the stream example for WASM.

Use these commands (command flow):

- emcmake cmake -S . -B build-wasm -DWHISPER_WASM_SINGLE_FILE=ON
- cmake --build build-wasm -j
or
- cmake --build build-wasm --target libstream -j$(nproc)

After build, copy from whisper.cpp/examples/stream.wasm/bin/*
- stream.js
- main.js
- libmain.js
- helpers.js

to the following directory:
shout/public/wasm/

## 2) Get Whisper model binaries

You can either:

- Download prebuilt GGML/GGUF-style Whisper binaries from a trusted source (for whisper.cpp), or
- Convert and quantize your own models with whisper.cpp

You can look at documentation or the included READMES in the repo

## 3) Model naming for app

Put model files in:

- shout/public/models/

The app resolves model URLs using these patterns (in order):

1. /models/whisper-{size}-{language}{q}.bin
2. /models/whisper-{size}.{language}{q}.bin
3. /models/whisper-{size}{q}.bin

Size is tiny or base (mobile uses tiny, desktop uses base),language is the selected language code (for example en), q is empty for full precision, or -q5 for quantized

Important:

- If a model path is wrong, dev server may return HTML instead of a model!!!!
- The worker guards against this by rejecting tiny/HTML responses, however! This may lead to an error if the model is less than 1024 (unlikely but just in case)

## 4) TODO: Seri and NXC LoRA adapters

LoRA adapter artifacts already exist in this repository:

- sei_ncx_asr_model/sei_lora_adapter/
- sei_ncx_asr_model/ncx_lora_adapter/

Problem is just we need to decide runtime target for them
  - Idea 1: server-side inference with adapter-enabled model
  - Idea 2: browser inference path

And also define conversion or export pipeline:
   - Merge adapter into a deployable model format
   - Serve adapter-aware model via backend endpoint

Need to add language-to-model routing:
   - en/fr -> current whisper.cpp browser models
   - sei/ncx -> adapter-backed inference route

And add evaluation:
   - Track WER/CER and morpheme-level quality
   - Compare baseline whisper.cpp vs adapter-backed output
   - Show this somewhere...
