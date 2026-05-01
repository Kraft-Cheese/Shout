# How to prepare data for training

## Download/Offline data
In *getData.py* data is downloaded for given speech data for OpenSLR/Commonvoice

Used also to load if offline (ie: Canada compute)

## Prepare Data
In *prepareData.py* prepare the datasets in with the following

## Dataset curation
- Parse Source metadata into paired samples
### CommonVoice
Read validated.tsv and map: clip path -> transcript -> speaker -> locale
### OpenSLR
Parse transcript files into same schema

## Standardize Audio Format
- Convert to 16 kHz, mono, PCM WAV

## Normalize transcripts
- Casing, punctuation, whitespace, Unicode

## Quality Filters
- Remove too short/ too long, empty, low SNR samples + reject log and reason

## Depduplicate

## Leakage-safe splits
- split by speaker/session if possible

## Generate training manifests
- train/val/test JSON: or CSV (audio_path, text, duration_sec, speaker_id, source, split)

## Dataset statistics & validation
- counts, hours, duration, speaker distribution

## Manifest Generation
