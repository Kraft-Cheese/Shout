# Download speech data from CommonVoice and from OpenSLR, and prepare it for training.
# These scripts should be downloaded and saved in a compressed format, and then extracted to save time and bandwidth.
import os
import shutil
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import requests

"""
How to use this script/ what it's for:
Set COMMON_VOICE_URL and/or the OPENSLR_URL
Note: this must be run first with a connection to the internet to download and prepare the data, and then it can be run without an internet connection to load the data from local storage.

The data is compressed and saved in the data/offline directory as speech_data.tar.gz, which can be transferred to other machines and extracted for use when there is no internet access. The script also provides functions to load the data from local storage for training.
To use this script the following libraries must be installed:
- requests
"""


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads"
EXTRACT_DIR = DATA_DIR / "extracted"
LOCAL_DATA_DIR = DATA_DIR / "local"
OFFLINE_DIR = DATA_DIR / "offline"
OFFLINE_ARCHIVE = OFFLINE_DIR / "speech_data.tar.gz"

def download_file(url, dest):
    """
    Download a file from a URL to a destination path.
    args: url to download, dest path to save the file
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with dest.open('wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def _is_safe_extract_path(base_dir, target_path):
    base_dir = Path(base_dir).resolve()
    target_path = Path(target_path).resolve()

    try:
        target_path.relative_to(base_dir)
        return True
    except ValueError:
        return False

def extract_tar_gz(file_path, extract_to):
    """
    Extract a .tar.gz file to a specified directory.
    args: file_path path to the .tar.gz file, extract_to directory to extract the contents
    """
    file_path = Path(file_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with tarfile.open(file_path, 'r:gz') as tar:
        for member in tar.getmembers():
            candidate_path = extract_to / member.name
            if not _is_safe_extract_path(extract_to, candidate_path):
                raise ValueError(f"Blocked unsafe tar entry: {member.name}")

        tar.extractall(path=extract_to)


def _archive_name_from_url(url, fallback_name):
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if name and name.endswith('.tar.gz'):
        return name
    return fallback_name

def prepare_common_voice_data(url=None):
    """Download and prepare CommonVoice data."""
    url = url or os.getenv("COMMON_VOICE_URL")
    if not url:
        raise ValueError("CommonVoice URL missing. Set COMMON_VOICE_URL or pass url.")

    dest = DOWNLOAD_DIR / _archive_name_from_url(url, "common_voice_data.tar.gz")
    extract_to = EXTRACT_DIR / "common_voice"

    print("Downloading CommonVoice data...")
    download_file(url, dest)
    print("Extracting CommonVoice data...")
    extract_tar_gz(dest, extract_to)
    print("CommonVoice data prepared.")
    return extract_to

def prepare_openslr_data(url=None):
    """Download and prepare OpenSLR data."""
    url = url or os.getenv("OPENSLR_URL")
    if not url:
        raise ValueError("OpenSLR URL missing. Set OPENSLR_URL or pass url.")

    dest = DOWNLOAD_DIR / _archive_name_from_url(url, "openslr_data.tar.gz")
    extract_to = EXTRACT_DIR / "openslr"

    print("Downloading OpenSLR data...")
    download_file(url, dest)
    print("Extracting OpenSLR data...")
    extract_tar_gz(dest, extract_to)
    print("OpenSLR data prepared.")
    return extract_to

# Save the necessary data files to a local directory
def save_data_locally(source_dirs=None, local_dir=LOCAL_DATA_DIR):
    """Save data locally for offline use."""
    if source_dirs is None:
        source_dirs = [EXTRACT_DIR / "common_voice", EXTRACT_DIR / "openslr"]

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    copied_paths = []
    for src in source_dirs:
        src = Path(src)
        if not src.exists():
            continue

        destination = local_dir / src.name
        if destination.exists():
            shutil.rmtree(destination)

        shutil.copytree(src, destination)
        copied_paths.append(destination)

    if not copied_paths:
        raise FileNotFoundError("No extracted datasets found to copy into local storage.")

    return copied_paths


# compress the data files into a format that can be easily transferred and extracted when there is no internet access
def compress_data_for_offline_use(source_dir=LOCAL_DATA_DIR, archive_path=OFFLINE_ARCHIVE):
    """Compress data for offline use."""
    source_dir = Path(source_dir)
    archive_path = Path(archive_path)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(source_dir, arcname=source_dir.name)

    return archive_path

# extract the compressed data files that were saved for offline use
def extract_data_for_offline_use(archive_path=OFFLINE_ARCHIVE, extract_to=DATA_DIR):
    """Extract data for offline use."""
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)

    if not archive_path.exists():
        raise FileNotFoundError(f"Offline archive not found: {archive_path}")

    extract_tar_gz(archive_path, extract_to)
    return extract_to / LOCAL_DATA_DIR.name


# load the necessary data files from a local directory
def load_data_locally(local_dir=LOCAL_DATA_DIR):
    """Load data from local storage."""
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local data directory not found: {local_dir}")

    datasets = {}
    for dataset_dir in sorted(local_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        file_count = sum(1 for p in dataset_dir.rglob('*') if p.is_file())
        datasets[dataset_dir.name] = {
            "path": str(dataset_dir),
            "file_count": file_count,
        }

    return datasets


def prepare_data_for_offline_use():
    """Prepare data for offline use."""
    copied_paths = save_data_locally()
    archive_path = compress_data_for_offline_use()
    return {
        "saved_paths": [str(p) for p in copied_paths],
        "archive": str(archive_path),
    }

def check_connection():
    """Check if there is an internet connection."""
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.RequestException:
        return False



if __name__ == "__main__":
    if check_connection():
        print("Internet connection detected.")

        downloaded_any = False

        if os.getenv("COMMON_VOICE_URL"):
            prepare_common_voice_data()
            downloaded_any = True
        else:
            print("Skipping CommonVoice download (COMMON_VOICE_URL not set).")

        if os.getenv("OPENSLR_URL"):
            prepare_openslr_data()
            downloaded_any = True
        else:
            print("Skipping OpenSLR download (OPENSLR_URL not set).")

        if downloaded_any:
            result = prepare_data_for_offline_use()
            loaded = load_data_locally()
            print(f"Offline archive created at: {result['archive']}")
            print(f"Local datasets: {', '.join(loaded.keys())}")
        else:
            print("No downloads performed. Set COMMON_VOICE_URL and/or OPENSLR_URL to fetch data.")
    else:
        print("No internet connection detected. Trying offline data...")
        try:
            extract_data_for_offline_use()
            loaded = load_data_locally()
            print(f"Loaded local datasets: {', '.join(loaded.keys())}")
        except FileNotFoundError as exc:
            print(f"Offline data unavailable: {exc}")
