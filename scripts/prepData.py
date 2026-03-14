"""
Prepare speech datasets (dowloaded from getData.py) for training

This script expects data produced by scripts/getData.py, then:
- Parses Common Voice and OpenSLR transcript metadata
- Normalizes text
- Converts audio to 16kHz mono PCM WAV
- Applies quality filters and deduplication
- Creates speaker-safe train/val/test splits
- Writes JSONL manifests, CSV summary, stats, and reject logs
"""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import unicodedata
import wave
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_LOCAL_DIR = DEFAULT_DATA_DIR / "local"
DEFAULT_EXTRACT_DIR = DEFAULT_DATA_DIR / "extracted"
DEFAULT_PREPARED_DIR = DEFAULT_DATA_DIR / "prepared"
FRONTEND_DATA_DIR = PROJECT_ROOT / "shout" / "data"

GET_DATA_MODULE = None

try:
	import getData as _get_data_module

	GET_DATA_MODULE = _get_data_module
except Exception:
	GET_DATA_MODULE = None


if GET_DATA_MODULE:
	DATA_DIR = Path(getattr(GET_DATA_MODULE, "DATA_DIR", DEFAULT_DATA_DIR))
	LOCAL_DATA_DIR = Path(getattr(GET_DATA_MODULE, "LOCAL_DATA_DIR", DEFAULT_LOCAL_DIR))
	EXTRACT_DIR = Path(getattr(GET_DATA_MODULE, "EXTRACT_DIR", DEFAULT_EXTRACT_DIR))
else:
	DATA_DIR = DEFAULT_DATA_DIR
	LOCAL_DATA_DIR = DEFAULT_LOCAL_DIR
	EXTRACT_DIR = DEFAULT_EXTRACT_DIR


AUDIO_EXTENSIONS = {
	".wav",
	".flac",
	".mp3",
	".m4a",
	".ogg",
	".opus",
	".aac",
	".sph",
}

INVALID_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class RawSample:
	source: str
	dataset: str
	audio_path: Path
	text: str
	speaker_id: str
	locale: str

# replace invalid filename characters with underscores, and trim leading/trailing dots and underscores
def slugify(value, default="unknown"):
	cleaned = INVALID_FILENAME_CHARS.sub("_", (value or "").strip())
	cleaned = cleaned.strip("._")
	return cleaned if cleaned else default

# apply Unicode NFKC normalization, collapse whitespace, and optionally lowercase and remove punctuation
def normalize_text(text, lowercase=True, remove_punctuation=False):
	if text is None:
		return ""

	normalized = unicodedata.normalize("NFKC", text)
	normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()

	if lowercase:
		normalized = normalized.lower()

	if remove_punctuation:
		normalized = re.sub(r"[^\w\s']", "", normalized)
		normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()

	return normalized

# gets the SHA-1 hash of a file's contents
def file_sha1(file_path):
	digest = hashlib.sha1()
	with Path(file_path).open("rb") as f:
		while True:
			chunk = f.read(1024 * 1024)
			if not chunk:
				break
			digest.update(chunk)
	return digest.hexdigest()

# gets the duration of a WAV file in seconds
def wav_duration_seconds(wav_path):
	with wave.open(str(wav_path), "rb") as wav_file:
		frame_rate = wav_file.getframerate()
		total_frames = wav_file.getnframes()
		if frame_rate <= 0:
			return 0.0
		return total_frames / float(frame_rate)

# detects "low-engery" audio ie: mostly silence or background noise, by calculating the mean absolute amplitude of PCM16 WAV data
def wav_mean_abs_amplitude(wav_path):
	"""Simple low-energy detector on PCM16 WAV data."""
	with wave.open(str(wav_path), "rb") as wav_file:
		if wav_file.getsampwidth() != 2:
			return 1000.0

		total_abs = 0
		sample_count = 0

		while True:
			frames = wav_file.readframes(4096)
			if not frames:
				break

			samples = array("h")
			samples.frombytes(frames)
			if sys.byteorder == "big":
				samples.byteswap()

			total_abs += sum(abs(sample) for sample in samples)
			sample_count += len(samples)

	if sample_count == 0:
		return 0.0

	return total_abs / float(sample_count)

def ffmpeg_available():
	return shutil.which("ffmpeg") is not None

# convert the audio files given the source and destination paths, using ffmpeg to convert to 16kHz mono PCM WAV
def convert_audio_to_wav(input_path, output_path):
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	command = [
		"ffmpeg",
		"-hide_banner",
		"-loglevel",
		"error",
		"-y",
		"-i",
		str(input_path),
		"-ac",
		"1",
		"-ar",
		"16000",
		"-c:a",
		"pcm_s16le",
		str(output_path),
	]
	subprocess.run(command, check=True)

# lookup audio files by utterance ID in OpenSLR datasets, checking common extensions and fallback patterns
def find_audio_by_utterance_id(directory, utterance_id):
	base = Path(directory)

	for extension in AUDIO_EXTENSIONS:
		candidate = base / f"{utterance_id}{extension}"
		if candidate.exists():
			return candidate

	for candidate in base.glob(f"{utterance_id}.*"):
		if candidate.suffix.lower() in AUDIO_EXTENSIONS:
			return candidate

	return None


def parse_common_voice(root_dir):
	samples = []
	parse_rejects = []
	root_dir = Path(root_dir)

	for tsv_path in sorted(root_dir.rglob("validated.tsv")):
		dataset_dir = tsv_path.parent
		clips_dir = dataset_dir / "clips"
		locale_hint = dataset_dir.name

		with tsv_path.open("r", encoding="utf-8-sig", newline="") as tsv_file:
			reader = csv.DictReader(tsv_file, delimiter="\t")

			has_required_columns = reader.fieldnames and "path" in reader.fieldnames and "sentence" in reader.fieldnames
			if not has_required_columns:
				parse_rejects.append(
					{
						"reason": "missing_required_columns",
						"source": "common_voice",
						"metadata_file": str(tsv_path),
					}
				)
				continue

			for row in reader:
				rel_audio = (row.get("path") or "").strip()
				text = (row.get("sentence") or "").strip()

				if not rel_audio:
					parse_rejects.append(
						{
							"reason": "missing_audio_reference",
							"source": "common_voice",
							"metadata_file": str(tsv_path),
							"text": text,
						}
					)
					continue

				audio_path = clips_dir / rel_audio
				if not audio_path.exists():
					audio_path = dataset_dir / rel_audio

				if not audio_path.exists():
					parse_rejects.append(
						{
							"reason": "audio_not_found",
							"source": "common_voice",
							"metadata_file": str(tsv_path),
							"audio_reference": rel_audio,
						}
					)
					continue

				samples.append(
					RawSample(
						source="common_voice",
						dataset=dataset_dir.name,
						audio_path=audio_path,
						text=text,
						speaker_id=(row.get("client_id") or row.get("speaker_id") or "unknown").strip() or "unknown",
						locale=(row.get("locale") or locale_hint).strip() or "unknown",
					)
				)

	return samples, parse_rejects


def parse_openslr(root_dir):
	samples = []
	parse_rejects = []
	root_dir = Path(root_dir)

	for trans_path in sorted(root_dir.rglob("*.trans.txt")):
		with trans_path.open("r", encoding="utf-8-sig") as trans_file:
			for line_number, raw_line in enumerate(trans_file, start=1):
				line = raw_line.strip()
				if not line:
					continue

				parts = line.split(maxsplit=1)
				if len(parts) != 2:
					parse_rejects.append(
						{
							"reason": "invalid_transcript_line",
							"source": "openslr",
							"metadata_file": str(trans_path),
							"line_number": line_number,
							"line": line,
						}
					)
					continue

				utterance_id, text = parts
				audio_path = find_audio_by_utterance_id(trans_path.parent, utterance_id)

				if not audio_path:
					parse_rejects.append(
						{
							"reason": "audio_not_found",
							"source": "openslr",
							"metadata_file": str(trans_path),
							"line_number": line_number,
							"audio_reference": utterance_id,
						}
					)
					continue

				speaker_id = utterance_id.split("-")[0] if "-" in utterance_id else "unknown"

				samples.append(
					RawSample(
						source="openslr",
						dataset=trans_path.parent.name,
						audio_path=audio_path,
						text=text,
						speaker_id=speaker_id or "unknown",
						locale="unknown",
					)
				)

	return samples, parse_rejects

# split the data from samples kept vs rejected during parsing
def collect_raw_samples(input_roots):
	all_samples = []
	all_parse_rejects = []

	for root in input_roots:
		cv_samples, cv_rejects = parse_common_voice(root)
		slr_samples, slr_rejects = parse_openslr(root)

		all_samples.extend(cv_samples)
		all_samples.extend(slr_samples)
		all_parse_rejects.extend(cv_rejects)
		all_parse_rejects.extend(slr_rejects)

	return all_samples, all_parse_rejects

def assign_speaker_safe_splits(prepared_samples, val_ratio, test_ratio, seed):
	train_ratio = 1.0 - val_ratio - test_ratio
	if train_ratio <= 0:
		raise ValueError("val_ratio + test_ratio must be less than 1.0")

	grouped = defaultdict(list)
	for sample in prepared_samples:
		grouped[sample["speaker_id"]].append(sample)

	speakers = list(grouped.keys())
	rng = random.Random(seed)
	rng.shuffle(speakers)
	speakers.sort(key=lambda speaker: len(grouped[speaker]), reverse=True)

	total_count = len(prepared_samples)
	targets = {
		"train": total_count * train_ratio,
		"val": total_count * val_ratio,
		"test": total_count * test_ratio,
	}
	counts = {"train": 0, "val": 0, "test": 0}
	speaker_split = {}

	for speaker in speakers:
		split = max(targets, key=lambda name: targets[name] - counts[name])
		speaker_split[speaker] = split
		counts[split] += len(grouped[speaker])

	split_samples = []
	for sample in prepared_samples:
		sample_copy = dict(sample)
		sample_copy["split"] = speaker_split[sample["speaker_id"]]
		split_samples.append(sample_copy)

	return split_samples


def build_output_audio_path(raw_sample, audio_dir):
	speaker = slugify(raw_sample.speaker_id, default="unknown")
	stem = slugify(raw_sample.audio_path.stem, default="audio")
	digest = hashlib.sha1(str(raw_sample.audio_path).encode("utf-8")).hexdigest()[:10]
	filename = f"{stem}_{digest}.wav"
	return Path(audio_dir) / raw_sample.source / speaker / filename


def process_samples(raw_samples, output_audio_dir, args):
	if not args.no_convert and not ffmpeg_available():
		raise RuntimeError(
			"ffmpeg is required for audio conversion. Install ffmpeg or use --no-convert."
		)

	output_audio_dir = Path(output_audio_dir)
	output_audio_dir.mkdir(parents=True, exist_ok=True)

	prepared = []
	rejects = []

	seen_audio_hashes = set()
	seen_text_keys = set()

	for raw_sample in raw_samples:
		normalized_text = normalize_text(
			raw_sample.text,
			lowercase=not args.keep_case,
			remove_punctuation=args.remove_punctuation,
		)

		if not normalized_text:
			rejects.append(
				{
					"reason": "empty_text_after_normalization",
					"source": raw_sample.source,
					"audio_path": str(raw_sample.audio_path),
				}
			)
			continue

		if args.no_convert:
			audio_out_path = raw_sample.audio_path
			if audio_out_path.suffix.lower() != ".wav":
				rejects.append(
					{
						"reason": "unsupported_audio_without_conversion",
						"source": raw_sample.source,
						"audio_path": str(raw_sample.audio_path),
					}
				)
				continue
		else:
			audio_out_path = build_output_audio_path(raw_sample, output_audio_dir)
			try:
				convert_audio_to_wav(raw_sample.audio_path, audio_out_path)
			except subprocess.CalledProcessError:
				rejects.append(
					{
						"reason": "ffmpeg_conversion_failed",
						"source": raw_sample.source,
						"audio_path": str(raw_sample.audio_path),
					}
				)
				continue

		try:
			duration_sec = wav_duration_seconds(audio_out_path)
		except wave.Error:
			rejects.append(
				{
					"reason": "invalid_wav",
					"source": raw_sample.source,
					"audio_path": str(audio_out_path),
				}
			)
			continue

		if duration_sec < args.min_duration_sec:
			rejects.append(
				{
					"reason": "too_short",
					"source": raw_sample.source,
					"audio_path": str(audio_out_path),
					"duration_sec": round(duration_sec, 4),
				}
			)
			continue

		if duration_sec > args.max_duration_sec:
			rejects.append(
				{
					"reason": "too_long",
					"source": raw_sample.source,
					"audio_path": str(audio_out_path),
					"duration_sec": round(duration_sec, 4),
				}
			)
			continue

		if not args.skip_low_energy_filter:
			mean_abs = wav_mean_abs_amplitude(audio_out_path)
			if mean_abs < args.min_mean_abs_amplitude:
				rejects.append(
					{
						"reason": "low_energy",
						"source": raw_sample.source,
						"audio_path": str(audio_out_path),
						"mean_abs_amplitude": round(mean_abs, 4),
					}
				)
				continue

		audio_hash = file_sha1(audio_out_path)
		if audio_hash in seen_audio_hashes:
			rejects.append(
				{
					"reason": "duplicate_audio",
					"source": raw_sample.source,
					"audio_path": str(audio_out_path),
				}
			)
			continue

		text_key = (raw_sample.speaker_id, normalized_text)
		if text_key in seen_text_keys:
			rejects.append(
				{
					"reason": "duplicate_text_for_speaker",
					"source": raw_sample.source,
					"audio_path": str(audio_out_path),
				}
			)
			continue

		seen_audio_hashes.add(audio_hash)
		seen_text_keys.add(text_key)

		prepared.append(
			{
				"audio_path": str(audio_out_path.resolve()),
				"text": normalized_text,
				"duration_sec": round(duration_sec, 4),
				"speaker_id": raw_sample.speaker_id or "unknown",
				"source": raw_sample.source,
				"dataset": raw_sample.dataset,
				"locale": raw_sample.locale,
			}
		)

	return prepared, rejects

# write a list of dictionaries to a JSONL file, ensuring the output directory exists
def write_jsonl(rows, output_path):
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")

# write a list of dictionaries to a CSV file, ensuring the output directory exists
def write_csv(rows, output_path, fieldnames):
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", encoding="utf-8", newline="") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)

# builds a statistics summary of the prepared samples and rejects, including counts, durations, splits, sources, and reject reasons
def build_stats(prepared_samples, rejects):
	stats = {
		"accepted_samples": len(prepared_samples),
		"rejected_samples": len(rejects),
		"total_hours": round(sum(s["duration_sec"] for s in prepared_samples) / 3600.0, 4),
		"splits": {},
		"sources": {},
	}

	for split_name in ("train", "val", "test"):
		split_rows = [s for s in prepared_samples if s.get("split") == split_name]
		stats["splits"][split_name] = {
			"count": len(split_rows),
			"hours": round(sum(s["duration_sec"] for s in split_rows) / 3600.0, 4),
			"unique_speakers": len({s["speaker_id"] for s in split_rows}),
		}

	source_groups = defaultdict(list)
	for sample in prepared_samples:
		source_groups[sample["source"]].append(sample)

	for source_name, rows in source_groups.items():
		stats["sources"][source_name] = {
			"count": len(rows),
			"hours": round(sum(s["duration_sec"] for s in rows) / 3600.0, 4),
			"unique_speakers": len({s["speaker_id"] for s in rows}),
		}

	reject_reasons = defaultdict(int)
	for reject in rejects:
		reject_reasons[reject.get("reason", "unknown")] += 1

	stats["reject_reasons"] = dict(sorted(reject_reasons.items(), key=lambda item: item[0]))
	return stats

# finds all audio files in a directory recursively, and copies them to a local directory for offline use, returning the list of copied paths
def discover_input_roots(user_input_dirs):
	resolved = []

	if user_input_dirs:
		for user_dir in user_input_dirs:
			path = Path(user_dir).expanduser().resolve()
			if path.exists():
				resolved.append(path)
		return resolved

	if LOCAL_DATA_DIR.exists() and any(LOCAL_DATA_DIR.iterdir()):
		return [LOCAL_DATA_DIR]

	if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
		return [EXTRACT_DIR]

	if FRONTEND_DATA_DIR.exists() and any(FRONTEND_DATA_DIR.iterdir()):
		return [FRONTEND_DATA_DIR]

	return []

# restores the data from a compressed archive for offline use, if the getData module is available and the archive exists, returning the path to the restored data or None if it cannot be restored
def try_restore_offline_data():
	if not GET_DATA_MODULE:
		return None

	restore_fn = getattr(GET_DATA_MODULE, "extract_data_for_offline_use", None)
	if not callable(restore_fn):
		return None

	try:
		restored_path = Path(restore_fn())
	except Exception:
		return None

	if restored_path.exists() and any(restored_path.iterdir()):
		return restored_path

	return None


def prepare_dataset(args):
	output_dir = Path(args.output_dir).expanduser().resolve()
	output_audio_dir = output_dir / "audio"
	manifests_dir = output_dir / "manifests"
	reports_dir = output_dir / "reports"

	input_roots = discover_input_roots(args.input_dir)

	if not input_roots:
		restored_path = try_restore_offline_data()
		if restored_path:
			input_roots = [restored_path]

	if not input_roots:
		raise FileNotFoundError(
			"No input data found. Run scripts/getData.py first or pass --input-dir."
		)

	print(f"Input roots: {', '.join(str(path) for path in input_roots)}")
	raw_samples, parse_rejects = collect_raw_samples(input_roots)

	if not raw_samples:
		raise RuntimeError("No speech samples parsed from input roots.")

	print(f"Parsed raw samples: {len(raw_samples)}")
	prepared_samples, prep_rejects = process_samples(raw_samples, output_audio_dir, args)

	all_rejects = parse_rejects + prep_rejects

	if not prepared_samples:
		raise RuntimeError("All samples were rejected during processing.")

	split_samples = assign_speaker_safe_splits(
		prepared_samples,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		seed=args.seed,
	)

	manifests_dir.mkdir(parents=True, exist_ok=True)
	reports_dir.mkdir(parents=True, exist_ok=True)

	for split_name in ("train", "val", "test"):
		split_rows = [row for row in split_samples if row["split"] == split_name]
		write_jsonl(split_rows, manifests_dir / f"{split_name}.jsonl")

	write_csv(
		split_samples,
		manifests_dir / "all_samples.csv",
		fieldnames=[
			"audio_path",
			"text",
			"duration_sec",
			"speaker_id",
			"source",
			"dataset",
			"locale",
			"split",
		],
	)

	write_csv(
		all_rejects,
		reports_dir / "reject_log.csv",
		fieldnames=sorted({key for row in all_rejects for key in row.keys()}) if all_rejects else ["reason"],
	)

	stats = build_stats(split_samples, all_rejects)
	with (reports_dir / "stats.json").open("w", encoding="utf-8") as stats_file:
		json.dump(stats, stats_file, indent=2, ensure_ascii=False)

	print("Prepared dataset artifacts:")
	print(f"- Manifests: {manifests_dir}")
	print(f"- Reports: {reports_dir}")
	print(f"- Converted audio: {output_audio_dir}")
	print(f"- Accepted samples: {stats['accepted_samples']}")
	print(f"- Rejected samples: {stats['rejected_samples']}")
	print(f"- Total hours: {stats['total_hours']}")


def build_arg_parser():
	parser = argparse.ArgumentParser(description="Prepare speech data for training.")

	parser.add_argument(
		"--input-dir",
		action="append",
		default=[],
		help="Input root directory. Repeat for multiple roots. Defaults to getData local/extracted paths.",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(DEFAULT_PREPARED_DIR),
		help="Output directory for prepared audio, manifests, and reports.",
	)

	parser.add_argument("--min-duration-sec", type=float, default=0.4, help="Minimum audio duration.")
	parser.add_argument("--max-duration-sec", type=float, default=30.0, help="Maximum audio duration.")
	parser.add_argument(
		"--val-ratio",
		type=float,
		default=0.1,
		help="Validation split ratio.",
	)
	parser.add_argument(
		"--test-ratio",
		type=float,
		default=0.1,
		help="Test split ratio.",
	)
	parser.add_argument("--seed", type=int, default=13, help="Random seed for split reproducibility.")

	parser.add_argument("--keep-case", action="store_true", help="Do not lowercase transcript text.")
	parser.add_argument(
		"--remove-punctuation",
		action="store_true",
		help="Remove punctuation after text normalization.",
	)
	parser.add_argument(
		"--no-convert",
		action="store_true",
		help="Skip ffmpeg conversion. Only WAV files are accepted in this mode.",
	)
	parser.add_argument(
		"--skip-low-energy-filter",
		action="store_true",
		help="Disable low-energy rejection.",
	)
	parser.add_argument(
		"--min-mean-abs-amplitude",
		type=float,
		default=50.0,
		help="Minimum mean absolute amplitude threshold for low-energy filtering.",
	)

	return parser


def main():
	parser = build_arg_parser()
	args = parser.parse_args()

	if args.val_ratio < 0 or args.test_ratio < 0:
		parser.error("Split ratios must be non-negative.")

	if args.val_ratio + args.test_ratio >= 1.0:
		parser.error("val_ratio + test_ratio must be less than 1.0")

	if args.min_duration_sec < 0 or args.max_duration_sec <= 0:
		parser.error("Duration thresholds must be positive.")

	if args.min_duration_sec >= args.max_duration_sec:
		parser.error("min-duration-sec must be smaller than max-duration-sec.")

	prepare_dataset(args)


if __name__ == "__main__":
	main()
