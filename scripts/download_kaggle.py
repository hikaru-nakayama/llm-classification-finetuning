import os
import subprocess
from pathlib import Path

COMPETITION = "llm-classification-finetuning"
DEFAULT_DOWNLOAD_DIR = "/content/kaggle_data"
DEFAULT_RAW_DIR = "/content/drive/MyDrive/llm-classification-finetuning/data/raw"

DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", DEFAULT_DOWNLOAD_DIR))
RAW_DIR = Path(os.getenv("RAW_DIR", DEFAULT_RAW_DIR))

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

subprocess.run(
    ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DOWNLOAD_DIR)],
    check=True,
)

subprocess.run(
    f"unzip -o {DOWNLOAD_DIR}/*.zip -d {DOWNLOAD_DIR}",
    shell=True,
    check=True,
)

for name in ["train.csv", "test.csv", "sample_submission.csv"]:
    src = DOWNLOAD_DIR / name
    dst = RAW_DIR / name
    subprocess.run(["cp", str(src), str(dst)], check=True)
    src.unlink()

for zip_file in DOWNLOAD_DIR.glob("*.zip"):
    zip_file.unlink()

print("download done")
