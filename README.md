# Shout
Post ASR Morphological Reconstruction for Polysynthetic Languages

## Why?
Many STT models fail at the "boundaries" of words in polysynthetic languages such as: Cree, Mohawk, or Blackfoot. As shown by research and projects from the FLAIR group.
The idea here is to use a base STT model to output garbled chunks and feed it's output to a browser based "Morphological Adapter"

## How? (Maybe)
I will try and train a small LoRA for a local model based on the grammatical rules of the target language/dialect.
The adapter should ideally "fix" the broken STT output in real-time within the browser.
This way for lower support languages we get a voice to text that goes from phonetics - > semantics

## TODO

[x] Vite/Preact setup + Comlinks
[] Make the componements usable
[] Get UI looking nice
[] Get Whisper.cpp working in browser
[] LoRA + Whisper for 'a' language
[] Reduce & Reconstruct
[] How will we estimate confidence?

Whisper.cpp -> WASM with Emscripten
```
git clone [whisper.cpp location]
cd whisper.cpp && mkdir build-em && cd build-em
emcmake cmake ..
make -j
```
Look into necessary flags

Maybe use whisper-web-transcriber instead? TBD

LoRA Whisper
(Official PEFT INT8 ASR guide)[https://huggingface.co/docs/peft/main/en/task_guides/int8-asr]
(Official training notebook)[https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb]
(PEFT repository) [https://github.com/huggingface/peft]

Note: We can't upload the trained models to hugging face, to look into

For reconstruction
if we can use the Alberta tools
(BK-Trees)[https://yomguithereal.github.io/mnemonist/] / (Tries)[https://www.npmjs.com/package/trie-search] for lexicon

Whisper.cpp (if w use it) exposes token probabilities, but need to extend WASM bindings TBD

Look into ONNX/Transformers.js (used previously with splat) look into

Might need (AudioWorklet)[https://github.com/aolsenjazz/libsamplerate-js] and OfflineAudioContext

(VAD)[https://github.com/ricky0123/vad] if we wanna go button less

Currently on splat: the whisper model already works I've added it as an example