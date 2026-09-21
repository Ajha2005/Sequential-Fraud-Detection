"""Phase 0: download the real IEEE-CIS Fraud Detection dataset from Kaggle.

Kaggle requires authentication, so this cannot run unattended in a sandboxed
CI/agent environment (there is no Kaggle account attached here) -- run it
yourself, once, with your own Kaggle API token:

    1. Create an API token at https://www.kaggle.com/settings -> "Create New Token".
       This downloads kaggle.json.
    2. Place it at ~/.kaggle/kaggle.json (chmod 600), or export
       KAGGLE_USERNAME / KAGGLE_KEY as environment variables.
    3. pip install kaggle   (already in requirements.txt)
    4. Accept the competition rules at
       https://www.kaggle.com/c/ieee-fraud-detection/rules (required once,
       in the browser, before the API will allow the download).
    5. python -m sfraud.data.download

This writes train_transaction.csv and train_identity.csv into data/raw/.
Everything downstream (merge.py, sequences.py, ...) auto-detects these files
and uses them in place of the synthetic fallback -- no other code changes
needed.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from sfraud.utils.config import load_config, resolve_path

COMPETITION = "ieee-fraud-detection"
REQUIRED_FILES = ["train_transaction.csv", "train_identity.csv"]


def already_downloaded(raw_dir: Path) -> bool:
    return all((raw_dir / f).exists() for f in REQUIRED_FILES)


def main() -> None:
    cfg = load_config()
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    if already_downloaded(raw_dir):
        print(f"Found existing raw files in {raw_dir}, nothing to do.")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as e:
        print(
            "Kaggle credentials not found. Set up ~/.kaggle/kaggle.json or "
            "KAGGLE_USERNAME/KAGGLE_KEY env vars, then re-run this script.\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except ImportError:
        print("The 'kaggle' package is not installed. Run: pip install kaggle", file=sys.stderr)
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading competition data for '{COMPETITION}' into {raw_dir} ...")
    api.competition_download_files(COMPETITION, path=str(raw_dir))

    zip_path = raw_dir / f"{COMPETITION}.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        zip_path.unlink()

    if already_downloaded(raw_dir):
        print("Download complete:", ", ".join(REQUIRED_FILES))
    else:
        print(
            "Download finished but expected files are missing -- check the "
            f"competition file list and {raw_dir} contents.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
