"""Download the permitted Titanic file and share one reproducible data split."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, is_zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
SPLIT_PATH = ROOT / "artifacts" / "split.json"
SEED = 42
VALIDATION_SIZE = 0.2
COLUMNS = {
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
}


def download_train():
    """Use Kaggle authentication"""
    if DATA_PATH.is_file():
        return DATA_PATH
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = [str(Path(sys.executable).with_name("kaggle")), "competitions",
               "download", "titanic", "-f", "train.csv", "-p", str(DATA_PATH.parent), "-q"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            "Kaggle download failed. Run `.venv/bin/kaggle auth login`, join the "
            "Titanic competition, and retry."
        )
    # Kaggle may return a CSV or ZIP; read only the requested member.
    for candidate in (DATA_PATH, DATA_PATH.with_suffix(".csv.zip")):
        if candidate.is_file() and is_zipfile(candidate):
            with ZipFile(candidate) as archive:
                contents = archive.read("train.csv")
            DATA_PATH.write_bytes(contents)
            if candidate != DATA_PATH:
                candidate.unlink()
            break
    if not DATA_PATH.is_file():
        raise RuntimeError("Kaggle completed without producing data/train.csv.")
    return DATA_PATH


def validate_data(frame):
    """Check the file contract; descriptive anomalies belong in the notebook."""
    if set(frame.columns) != COLUMNS:
        raise ValueError("Expected the original Titanic train.csv columns.")
    if frame.empty or frame["PassengerId"].isna().any() or not frame["PassengerId"].is_unique:
        raise ValueError("PassengerId must be present and unique in a nonempty dataset.")
    if frame["Survived"].isna().any() or set(frame["Survived"].unique()) != {0, 1}:
        raise ValueError("Survived must contain both binary labels, 0 and 1, without missing values.")
    return frame


def load_data():
    return validate_data(pd.read_csv(download_train()))


def split_data(frame):
    """Return development and final validation data; record the exact membership."""
    validate_data(frame)
    development, validation = train_test_split(
        frame, test_size=VALIDATION_SIZE, stratify=frame["Survived"], random_state=SEED
    )
    receipt = {
        "source": "https://www.kaggle.com/competitions/titanic/data",
        "file": "train.csv",
        "data_sha256": hashlib.sha256(
            frame.sort_values("PassengerId").to_csv(index=False).encode()
        ).hexdigest(),
        "seed": SEED,
        "validation_fraction": VALIDATION_SIZE,
        "development_ids": sorted(development["PassengerId"].tolist()),
        "validation_ids": sorted(validation["PassengerId"].tolist()),
    }
    if SPLIT_PATH.exists() and json.loads(SPLIT_PATH.read_text()) != receipt:
        raise ValueError("Data or split settings differ from artifacts/split.json. Review before replacing it.")
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return development.copy(), validation.copy()


if __name__ == "__main__":
    development, validation = split_data(load_data())
    print(f"Saved split: {len(development)} development / {len(validation)} validation passengers.")
