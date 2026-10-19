#!/usr/bin/env python3
"""
Build outputs/colab_dev_package.zip for Google Colab GPU Execution.
Includes:
  - notebooks/colab_gpu_runner.ipynb
  - data/raw_scans/dev/ (p001.png to p011.png)
  - data/preprocessed_scans/dev/ (p001.png to p011.png)
  - data/gold_standard/dev/ (p001_gold.json to p011_gold.json)
  - outputs/ocr/gemini_raw/ (p001.json to p011.json)
  - outputs/ocr/tesseract_raw/ (p001.json to p011.json)
  - outputs/chunks/ (S1, S2, S3 chunks)
  - src/ (all source modules)
  - configs/ and data/pages.csv
"""

import os
from pathlib import Path
import zipfile

def build_package():
    root_dir = Path(__file__).resolve().parent.parent
    zip_out = root_dir / "outputs" / "colab_dev_package.zip"
    zip_out.parent.mkdir(parents=True, exist_ok=True)

    include_patterns = [
        "notebooks/colab_gpu_runner.ipynb",
        "data/pages.csv",
        "requirements.txt",
        "configs/**/*",
        "src/**/*.py",
        "data/raw_scans/dev/p*.png",
        "data/preprocessed_scans/dev/p*.png",
        "data/gold_standard/dev/p*.json",
        "outputs/ocr/gemini_raw/p*.json",
        "outputs/ocr/tesseract_raw/p*.json",
        "outputs/chunks/**/*.json"
    ]

    files_to_zip = []
    for pattern in include_patterns:
        matches = list(root_dir.glob(pattern))
        for m in matches:
            if m.is_file() and "__pycache__" not in str(m):
                files_to_zip.append(m)

    # Deduplicate
    files_to_zip = sorted(list(set(files_to_zip)))
    print(f"Total files to package: {len(files_to_zip)}")

    if zip_out.exists():
        zip_out.unlink()

    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_zip:
            arcname = f.relative_to(root_dir)
            zf.write(f, str(arcname))

    size_mb = zip_out.stat().st_size / (1024 * 1024)
    print(f"Successfully created: {zip_out} ({size_mb:.2f} MB, {len(files_to_zip)} files)")

if __name__ == "__main__":
    build_package()
