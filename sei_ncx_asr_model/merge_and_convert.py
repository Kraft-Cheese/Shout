#!/usr/bin/env python3
"""
Merges each language's LoRA adapter into whisper-small base model,
converts to GGML format, quantizes to Q5_0, and places the outputs in
public/models/ ready for the browser app.

Output files (placed in PUBLIC_MODELS_DIR):
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR       = Path(__file__).parent.resolve()
REPO_ROOT        = SCRIPT_DIR.parent.parent          # NLP/
WHISPER_CPP_DIR  = REPO_ROOT / "whisper.cpp"
CONVERT_SCRIPT   = WHISPER_CPP_DIR / "models" / "convert-h5-to-ggml.py"
QUANTIZE_BIN     = WHISPER_CPP_DIR / "build-native" / "bin" / "whisper-quantize"
PUBLIC_MODELS_DIR = SCRIPT_DIR.parent / "shout" / "public" / "models"

# Add adapter for new languages
ADAPTERS = {
    "sei": SCRIPT_DIR / "sei_lora_adapter",
    "ncx": SCRIPT_DIR / "ncx_lora_adapter",
}

QUANT_TYPE = "q5_0"

# mel_filters.npz is a static 200 KB file bundled with openai-whisper.
# Cache a local copy here so the script runs fully offline after first use.
MEL_FILTERS_CACHE = SCRIPT_DIR / "assets" / "mel_filters.npz"

# check for required dependencies and files
def check_deps():
    missing = []
    for pkg in ("transformers", "peft", "safetensors", "torch", "numpy"):
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if missing:
        print(f"[error] Missing Python packages: {', '.join(missing)}")
        print("        pip install transformers peft safetensors torch librosa")
        sys.exit(1)

    if not CONVERT_SCRIPT.exists():
        print(f"[error] convert-h5-to-ggml.py not found at {CONVERT_SCRIPT}")
        sys.exit(1)

    if not QUANTIZE_BIN.exists():
        print(f"[error] whisper-quantize not found at {QUANTIZE_BIN}")
        print("        Build whisper.cpp natively: cmake -B build-native && cmake --build build-native")
        sys.exit(1)

    PUBLIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# mel is a fixed filterbank based on the sample rate and FFT size, so we can generate it once and reuse it for all conversions
def ensure_mel_filters() -> Path:
    """
    Return a path to mel_filters.npz, generating and caching it if needed.

    Priority:
      1. Local cache (sei_ncx_asr_model/assets/mel_filters.npz) (fully offline)
      2. openai-whisper package assets (copy to cache)
      3. Compute from librosa (compute and cache)
    """

    # Check local cache first (should be fully offline after first run)
    if MEL_FILTERS_CACHE.exists():
        return MEL_FILTERS_CACHE

    # check if exists
    MEL_FILTERS_CACHE.parent.mkdir(parents=True, exist_ok=True)

    # Try openai-whisper package first
    if importlib.util.find_spec("whisper") is not None:
        import whisper as _w
        pkg_mel = Path(_w.__file__).parent / "assets" / "mel_filters.npz"
        if pkg_mel.exists():
            shutil.copy2(pkg_mel, MEL_FILTERS_CACHE)
            print(f"[mel_filters] Cached from openai-whisper → {MEL_FILTERS_CACHE}")
            return MEL_FILTERS_CACHE

    # Fall back to computing with librosa (matches whisper's exact filterbank)
    if importlib.util.find_spec("librosa") is None:
        print("[error] mel_filters.npz not found and neither openai-whisper nor librosa is installed.")
        print("        pip install openai-whisper   (or)   pip install librosa")
        sys.exit(1)

    import librosa
    import numpy as np

    # The mel filterbank is determined by the sample rate (16 kHz) and FFT size (400 for 25ms windows with 16000 Hz)
    print("[mel_filters] Computing mel filterbanks with librosa (16kHz, n_fft=400)...")
    filters_80  = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
    filters_128 = librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)
    np.savez(str(MEL_FILTERS_CACHE), mel_80=filters_80, mel_128=filters_128)
    print(f"[mel_filters] Saved to {MEL_FILTERS_CACHE} (will be reused offline)")
    return MEL_FILTERS_CACHE

# expects the same mel_filters.npz structure as openai-whisper, which is what convert-h5-to-ggml.py needs
def find_whisper_repo_root() -> Path:
    """
    The convert script expects <root>/whisper/assets/mel_filters.npz.
    We satisfy this by creating a temporary directory with that structure
    pointing at our cached mel_filters.npz.
    """
    mel_path = ensure_mel_filters()
    # Build a minimal directory tree that matches what convert-h5-to-ggml.py needs
    tmp_root = mel_path.parent.parent / "_whisper_root"
    assets_dir = tmp_root / "whisper" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    target = assets_dir / "mel_filters.npz"
    if not target.exists():
        shutil.copy2(mel_path, target)
    return tmp_root

# Merge LoRA adapter weights into the base model and save the merged model to out_dir
def merge_lora(language: str, adapter_dir: Path, out_dir: Path, base_model_path: str = "openai/whisper-small"):
    print(f"\n[{language}] Loading base model {base_model_path} ...")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    kwargs = {"local_files_only": True} if Path(base_model_path).exists() else {}

    # Load base model and processor (tokenizer + feature extractor)
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_path, **kwargs)
    processor  = WhisperProcessor.from_pretrained(base_model_path, **kwargs)

    print(f"[{language}] Loading LoRA adapter from {adapter_dir} ...")
    # PeftModel will automatically find the base model's weights and merge the adapter
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print(f"[{language}] Merging adapter weights ...")
    # Apply the LoRA weights to the base model and unload the adapter to free memory
    model = model.merge_and_unload()

    print(f"[{language}] Saving merged model to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save the merged model and processor (tokenizer + feature extractor) to the same directory
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))

    # convert-h5-to-ggml.py reads config.json directly
    config_path = out_dir / "config.json"
    if not config_path.exists():
        print(f"[error] config.json not written to {out_dir}")
        sys.exit(1)

    # Patch max_length if missing (older whisper-small configs omit it)
    with open(config_path) as f:
        cfg = json.load(f)
    if "max_length" not in cfg or cfg["max_length"] is None:
        cfg["max_length"] = cfg.get("max_target_positions", 448)
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[{language}] Patched max_length={cfg['max_length']} in config.json")

    print(f"[{language}] Merge done.")

# Convert the merged model to GGML format
def convert_to_ggml(language: str, merged_dir: Path, out_dir: Path, whisper_root: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp16_bin = out_dir / "ggml-model.bin"

    print(f"\n[{language}] Converting to GGML (F16) ...")

    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(merged_dir),
        str(whisper_root),
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[error] convert-h5-to-ggml.py failed:\n{result.stderr}")
        sys.exit(1)

    if not fp16_bin.exists():
        print(f"[error] Expected output {fp16_bin} not created.")
        sys.exit(1)

    size_mb = fp16_bin.stat().st_size / 1024 / 1024
    print(f"[{language}] GGML F16 written: {fp16_bin} ({size_mb:.0f} MB)")
    return fp16_bin

# Quantize the GGML model to the specified quantization type
def quantize(language: str, fp16_bin: Path, out_path: Path):
    print(f"\n[{language}] Quantizing to {QUANT_TYPE} to {out_path.name} ...")
    cmd = [str(QUANTIZE_BIN), str(fp16_bin), str(out_path), QUANT_TYPE]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[error] whisper-quantize failed:\n{result.stderr}")
        sys.exit(1)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[{language}] Quantized model written: {out_path} ({size_mb:.0f} MB)")

# Install the generated models to the public/models/ directory for use in the browser app
def install(language: str, fp16_bin: Path, quant_bin: Path, skip_fp16: bool):
    fp16_dest  = PUBLIC_MODELS_DIR / f"whisper-small-{language}.bin"
    quant_dest = PUBLIC_MODELS_DIR / f"whisper-small-{language}-q5.bin"

    if not skip_fp16:
        shutil.copy2(fp16_bin, fp16_dest)
        size_mb = fp16_dest.stat().st_size / 1024 / 1024
        print(f"[{language}] Installed {fp16_dest.name} ({size_mb:.0f} MB)")

    shutil.copy2(quant_bin, quant_dest)
    size_mb = quant_dest.stat().st_size / 1024 / 1024
    print(f"[{language}] Installed {quant_dest.name} ({size_mb:.0f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and convert to GGML")
    parser.add_argument("--languages", nargs="+", default=["sei", "ncx"],
                        choices=list(ADAPTERS.keys()),
                        help="Languages to process (default: sei ncx)")
    parser.add_argument("--skip-fp16", action="store_true",
                        help="Skip copying the large full-precision model to public/models/")
    parser.add_argument("--base-model", default="openai/whisper-small",
                        help="HuggingFace model ID or local path to whisper-small weights "
                             "(use a local path for fully offline operation)")
    args = parser.parse_args()

    check_deps()
    whisper_root = find_whisper_repo_root()
    print(f"Using mel_filters from: {whisper_root}/whisper/assets/mel_filters.npz")
    print(f"Output directory: {PUBLIC_MODELS_DIR}")

    for language in args.languages:
        adapter_dir = ADAPTERS[language]
        if not adapter_dir.exists():
            print(f"[error] Adapter not found: {adapter_dir}")
            sys.exit(1)

        with tempfile.TemporaryDirectory(prefix=f"shout-merge-{language}-") as tmp:
            tmp_path   = Path(tmp)
            merged_dir = tmp_path / "merged"
            ggml_dir   = tmp_path / "ggml"

            merge_lora(language, adapter_dir, merged_dir, base_model_path=args.base_model)
            fp16_bin  = convert_to_ggml(language, merged_dir, ggml_dir, whisper_root)
            quant_bin = ggml_dir / "ggml-model-q5.bin"
            quantize(language, fp16_bin, quant_bin)
            install(language, fp16_bin, quant_bin, skip_fp16=args.skip_fp16)

    print("\n[done] All models installed to public/models/")
    print("Update whisper.workerThread.js model routing for sei/ncx to whisper-small")


if __name__ == "__main__":
    main()
