# Shout

Post-ASR, confidence-based lexical reconstruction for low-resource polysynthetic
languages

Currently Supported Languages:
```
(Seri, `sei`; Central Puebla Nahuatl, `ncx`).
```

## Why

Off-the-shelf speech-to-text models tend to fail at word boundaries in
polysynthetic languages, where a single utterance can encode what English
expresses with a full sentence. Shout pairs a base STT model (Whisper) with a
lightweight reconstruction step that repairs the broken output, and runs the
whole pipeline locally in the browser via WebAssembly so that audio never
leaves the user's device.

## Repository layout

```
Shout/
  README.md                 this file! ;D
  requirements.txt          Python dependencies for the experiments
  splat.html                Reference past project (ONNX-based, for inspiration only)
  scripts/                  data download and preparation utilities
    getData.py
    prepData.py
    DataPrep.md
  whisper_files/            replacement files for the whisper.cpp WASM build
    CMakeLists.txt
    emscripten.cpp
  shout/                    Preact + Vite browser frontend (Preact == lightweight React)
    src/                    components, worker bridge, BK-tree reconstruction
    public/wasm/            compiled whisper.cpp WebAssembly bundle
    public/models/          ggml-format Whisper binaries served by Vite
    public/vocab/           per-language wordlists used by reconstruction
    package.json
    README.md               frontend-specific build notes
  sei_ncx_asr_model/        Python ML experiments (LoRA fine-tuning + evaluation)
    eval_b1_zero_shot.py    B1: Whisper zero-shot baseline
    whisper_lora_train.py   B2: Whisper + LoRA fine-tuning
    reconstruct_b4.py       B4: lexicon-constrained reconstruction
    eval_morpheme_f1.py     morpheme/boundary F1 evaluation
    merge_and_convert.py    merge LoRA into base, convert to ggml, quantise
    sei_lora_adapter/       trained LoRA adapter for Seri
    ncx_lora_adapter/       trained LoRA adapter for Nahuatl
    b1_results/             zero-shot baseline predictions and metrics
    b4_results/             reconstruction predictions and metrics
    EXPERIMENT_NOTES.md
```

## Reproducing the experiments

### Python version

The experiments were run with **Python 3.10**. Anything 3.10 or newer should
work; the scripts use `tuple[...]` generics and modern `transformers`/`peft`
APIs, so 3.9 or older is not supported.

### Required libraries

A `requirements.txt` is provided at the repository root. To recreate the
environment in a fresh virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```
_Note: I highly reccomend you run this project on WSL or Linux, I have not tested from windows_


The pinned versions reflect the environment used to produce the results in
`sei_ncx_asr_model/b1_results/` and `sei_ncx_asr_model/b4_results/`. To
regenerate the file from your own environment after upgrades, use:

```bash
python -m pip freeze > requirements.txt
```

### Data

`scripts/getData.py` downloads Common Voice and/or OpenSLR archives, extracts
them, and (optionally) packages them for offline use. Set the source URLs as
environment variables and run the script:

```bash
export COMMON_VOICE_URL="https://.../cv-corpus-XX-sei.tar.gz"
export OPENSLR_URL="https://.../openslr-resource.tar.gz"
python scripts/getData.py
```
_Note: They are big! Make sure you have enough memory beforehand_

`scripts/prepData.py` and `scripts/DataPrep.md` document the standardisation
steps: 16 kHz mono WAV, transcript normalisation, quality filters,
deduplication, and train/dev/test splits.

_Note: These are all standard reccomendations for how to clean and normalize data when using wWhisper and CommonVoice_

### Running the baselines

From `sei_ncx_asr_model/`:

```bash
# B1 ie: zero-shot Whisper
python eval_b1_zero_shot.py --language sei
python eval_b1_zero_shot.py --language ncx

# B2 ie: Whisper + LoRA fine-tuning (saves adapter only (merge_and_convert later!), ~50 MB)
python whisper_lora_train.py --language sei
python whisper_lora_train.py --language ncx

# B4 ie: uses the lexicon reconstruction over B1 or B2 predictions
python reconstruct_b4.py \
  --predictions b1_results/b1_zero_shot_sei_predictions.csv \
  --train_tsv /path/to/cv/sei/train.tsv \
  --language sei

# Boundary and Token F1 evaluations (before vs. after reconstruction)
python eval_morpheme_f1.py --lang sei
python eval_morpheme_f1.py --lang ncx
```

The training script expects Common Voice data in one of the locations listed
in `find_existing_cv_data()`; pass `--data_dir` to override.

### Current results

| Language | WER (B1) | WER (B4) | Token F1 (B1 to B4) | Boundary F1 (B1 to B4) |
|----------|----------|----------|--------------------|-----------------------|
| sei      | 1.35     | 1.34     | 0.010 => 0.028      | 0.140 => 0.146         |
| ncx      | 1.78     | 1.68     | 0.066 => 0.156      | 0.196 => 0.224         |

B4 reconstruction uses a Levenshtein lookup against the training lexicon with
a maximum edit distance of 2. No regressions were observed for `sei`; two
regressions were observed for `ncx`.

## Browser frontend

The frontend lives in `Shout/shout/` and is a Preact + Vite app. It loads a
quantised Whisper model via a whisper.cpp WebAssembly build running in a Web
Worker, and post-processes the output with an in-browser BK-tree reconstruction
step (`src/lib/reconstruct.js`). See `shout/README.md` for the full build
instructions; the short version follows.

### 1. Build whisper.cpp for WebAssembly

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
```

Replace `examples/stream.wasm/CMakeLists.txt` and
`examples/stream.wasm/emscripten.cpp` with the versions in
`Shout/whisper_files/`, then build:

```bash
emcmake cmake -S . -B build-wasm -DWHISPER_WASM_SINGLE_FILE=ON
cmake --build build-wasm -j
```

Copy the resulting `stream.js`, `main.js`, `libmain.js`, and `helpers.js` from
`build-wasm/bin/stream.wasm/` into `Shout/shout/public/wasm/`.
(These files are already copied in for convenience. But if ever you which to adapt htis project this would be how! :D)

### 2. Producing the per-language model binaries

The browser worker expects ggml-format Whisper binaries in
`shout/public/models/`. The repository does not ship `whisper-base.en.bin` and
`whisper-base.en-q5.bin` for the English fallback path. These MUST be sourced from Whisper.cpp project itself and copied over, or in releases. The Seri and Nahuatl
binaries are built from the LoRA adapters in `sei_ncx_asr_model/` because
they are too large to commit and depend the a local whisper.cpp checkout (Recommened anyways).

The end-to-end pipeline for one language is:

1. **Train a LoRA adapter** with `whisper_lora_train.py --language <lang>`
   (or use the adapters already saved in `sei_lora_adapter/` and
   `ncx_lora_adapter/`). This produces a small (~50 MB) PEFT adapter on top
   of `openai/whisper-small`. (You can change the size of whisper if you wish, up to you, but just be concious of browser constraints)
2. **Merge, convert, and quantise** the adapter into a single ggml binary
   that whisper.cpp can load:

   ```bash
   cd sei_ncx_asr_model
   python merge_and_convert.py
   ```

   `merge_and_convert.py` looks at the `ADAPTERS` dictionary at the top of the
   script, merges each adapter into the base `whisper-small` weights,
   invokes `whisper.cpp/models/convert-h5-to-ggml.py` to produce a ggml
   file, runs `whisper.cpp/build-native/bin/whisper-quantize` with `q5_0`,
   and copies the result into `shout/public/models/` under the naming
   convention the worker expects. If you want to add an adapter you need to update this script!

To run step 2 you need a sibling `whisper.cpp` (told you it would be useful!) checkout next to `Shout/`,
with the converter script available and a native (non-WASM) build of
`whisper-quantize`:

```bash
# from the parent directory of Shout/
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -S . -B build-native
cmake --build build-native --target whisper-quantize -j
```

After this, `merge_and_convert.py` is the single command that turns a
trained adapter into a browser-ready model file. Adding a new language is a
matter of training a new LoRA, registering it in the `ADAPTERS` dict, and
re-running the script.

The worker resolves model URLs in this order:

1. `/models/whisper-{size}-{language}{q}.bin`
2. `/models/whisper-{size}.{language}{q}.bin`
3. `/models/whisper-{size}{q}.bin`
4. `/models/whisper-base.en{q}.bin` (English fallback)
5. `/models/whisper-base{q}.bin`

`size` is `tiny` or `base` for generic languages (mobile uses `tiny`) and
`small` for `sei`/`ncx` (LoRA-fine-tuned). `q` is empty for full precision or
`-q5` for the quantised build. Models are cached via the Cache API for
offline reuse on subsequent loads.

### 3. Run the app

```bash
cd shout
npm install
npm run dev          # development server
# or
npm run build && npm run preview
```

## Acknowledgements

The browser ASR architecture is inspired by the splat.html demo from
Integrated AI Systems (`splat.html` in this repo), which uses ONNX rather
than a compiled whisper.cpp build.
