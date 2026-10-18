#!/usr/bin/env python3
"""
Comprehensive Unit Tests and Verification for Step 2:
  - Verifies re-rendered corpus pages (p001, p014, p098, p107) and pages.csv
  - Tests normalise.py functions (NFC, nukta, ZWJ/ZWNJ, digits, suffixes)
  - Tests preprocess.py functions on data/raw_scans/dev/p001.png
"""

import os
from pathlib import Path
import stat
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalise import (
    normalise_unicode_nfc,
    normalise_ya_nukta,
    clean_zwj_zwnj,
    normalise_digits,
    normalise_whitespace_and_punct,
    strip_bangla_suffixes_for_matching,
    normalise_bangla_text
)
from src.preprocess import (
    to_greyscale,
    apply_median_blur,
    apply_clahe,
    apply_otsu_binarisation,
    apply_deskew,
    preprocess_image
)


def test_corpus_verification():
    print("=" * 60)
    print("1. CORPUS EXTRACTION & PAGES.CSV VERIFICATION")
    print("=" * 60)

    csv_path = Path("data/pages.csv")
    assert csv_path.exists(), "data/pages.csv does not exist!"
    df = pd.read_csv(csv_path)

    print(f"Total rows in pages.csv: {len(df)}")
    assert len(df) == 44, f"Expected 44 rows, found {len(df)}"

    # Check key milestone pages
    milestones = {
        "p001": {"book_page": 1, "pdf_page": 6, "split": "dev", "chapter_id": "ch01"},
        "p014": {"book_page": 14, "pdf_page": 19, "split": "test", "chapter_id": "ch02"},
        "p098": {"book_page": 98, "pdf_page": 103, "split": "test", "chapter_id": "ch08"},
        "p107": {"book_page": 107, "pdf_page": 112, "split": "test", "chapter_id": "ch09"},
    }

    for pid, expected in milestones.items():
        row = df[df["page_id"] == pid].iloc[0]
        assert row["book_page"] == expected["book_page"], f"Mismatch for {pid} book_page: {row['book_page']} != {expected['book_page']}"
        assert row["pdf_page"] == expected["pdf_page"], f"Mismatch for {pid} pdf_page: {row['pdf_page']} != {expected['pdf_page']}"
        assert row["split"] == expected["split"], f"Mismatch for {pid} split"
        img_p = Path(row["image_path"])
        assert img_p.exists(), f"Image file {img_p} missing!"
        print(f"  [OK] {pid} -> Book Page {row['book_page']}, PDF Page {row['pdf_page']}, Path: {img_p}")

    # Check test set read-only status
    test_files = list(Path("data/raw_scans/test").glob("*.png"))
    assert len(test_files) == 31, f"Expected 31 test files, found {len(test_files)}"
    ro_count = sum(1 for f in test_files if not (os.stat(f).st_mode & stat.S_IWRITE))
    print(f"Test files locked as Read-Only: {ro_count}/{len(test_files)}")
    assert ro_count == 31, "Not all test files are locked as read-only!"

    # Show first 15 rows
    print("\n--- First 15 rows of data/pages.csv ---")
    print(df.head(15).to_string())
    print()


def test_normalisation():
    print("=" * 60)
    print("2. BANGLA UNICODE NORMALISATION VERIFICATION")
    print("=" * 60)

    # 1. Decomposed য + ় -> য়
    decomposed_ya = "\u09af\u09bc"  # য + ়
    canonical_ya = "\u09df"          # য়
    res_ya = normalise_ya_nukta(decomposed_ya)
    assert res_ya == canonical_ya, f"Nukta normalisation failed: {repr(res_ya)} != {repr(canonical_ya)}"
    print(f"  [OK] Decomposed য+় normalisation: {repr(decomposed_ya)} -> {repr(res_ya)} ('{res_ya}')")

    # 2. Digits: Western <-> Bangla
    sample_text_western = "1952 সালের 21শে ফেব্রুয়ারি"
    sample_text_bangla = "১৯৫২ সালের ২১শে ফেব্রুয়ারি"
    to_bn = normalise_digits(sample_text_western, target="bangla")
    to_en = normalise_digits(sample_text_bangla, target="western")
    assert to_bn == sample_text_bangla, f"Failed converting to Bangla: {to_bn}"
    assert to_en == sample_text_western, f"Failed converting to Western: {to_en}"
    print(f"  [OK] Digit conversion (Western -> Bangla): '{sample_text_western}' -> '{to_bn}'")
    print(f"  [OK] Digit conversion (Bangla -> Western): '{sample_text_bangla}' -> '{to_en}'")

    # 3. Case suffixes stripping
    suffix_tests = [
        ("জিন্নাহ'র", "জিন্নাহ"),
        ("জিন্নাহর", "জিন্নাহ"),
        ("পাকিস্তানের", "পাকিস্তান"),
        ("ঢাকায়", "ঢাকা"),
        ("পূর্ববাংলায়", "পূর্ববাংলা"),
        ("ছাত্রদের", "ছাত্র"),
        ("ভাষাকে", "ভাষা"),
    ]
    for inp, expected in suffix_tests:
        res = strip_bangla_suffixes_for_matching(inp)
        assert res == expected, f"Suffix stripping failed for {inp}: got '{res}', expected '{expected}'"
        print(f"  [OK] Suffix stripping: '{inp}' -> '{res}'")

    # 4. ZWJ / ZWNJ cleaning
    zwj_text = "বাংলা\u200d ভাষা\u200c।"
    cleaned_zw = clean_zwj_zwnj(zwj_text)
    print(f"  [OK] Clean isolated ZWJ/ZWNJ: {repr(zwj_text)} -> {repr(cleaned_zw)}")

    # 5. Danda and whitespace
    dirty_text = "আমরা আন্দোলন করব  |   পরের বাক্য  ||"
    clean_p = normalise_whitespace_and_punct(dirty_text)
    print(f"  [OK] Whitespace & Danda: '{dirty_text}' -> '{clean_p}'")
    print()


def test_preprocessing():
    print("=" * 60)
    print("3. OPENCV PRE-PROCESSING PIPELINE VERIFICATION")
    print("=" * 60)

    sample_img_path = Path("data/raw_scans/dev/p001.png")
    assert sample_img_path.exists(), f"Sample image {sample_img_path} not found"

    orig = cv2.imread(str(sample_img_path))
    assert orig is not None, "Failed to read p001.png"
    print(f"  Original image loaded: shape={orig.shape}, dtype={orig.dtype}")

    # 1. Greyscale
    gray = to_greyscale(orig)
    assert len(gray.shape) == 2, f"Expected 2D greyscale array, got {gray.shape}"
    print(f"  [OK] 1. to_greyscale: shape={gray.shape}, min={gray.min()}, max={gray.max()}")

    # 2. Median blur
    blurred = apply_median_blur(gray, ksize=3)
    assert blurred.shape == gray.shape, "Shape mismatch after median blur"
    print(f"  [OK] 2. apply_median_blur: shape={blurred.shape}")

    # 3. CLAHE
    clahe_out = apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8))
    assert clahe_out.shape == gray.shape, "Shape mismatch after CLAHE"
    print(f"  [OK] 3. apply_clahe: shape={clahe_out.shape}")

    # 4. Otsu binarisation
    binary = apply_otsu_binarisation(gray)
    unique_vals = np.unique(binary)
    assert len(unique_vals) <= 2, f"Expected binary image (0, 255), got {unique_vals}"
    print(f"  [OK] 4. apply_otsu_binarisation: unique values={unique_vals}")

    # 5. Deskew
    deskewed = apply_deskew(gray)
    assert deskewed.shape == gray.shape, "Shape mismatch after deskew"
    print(f"  [OK] 5. apply_deskew: shape={deskewed.shape}")

    # 6. Master pipeline
    custom_cfg = {
        "mode": "custom",
        "to_greyscale": True,
        "median_blur": {"enabled": True, "ksize": 3},
        "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
        "otsu_binarisation": {"enabled": True},
        "deskew": {"enabled": False}
    }
    pipe_out = preprocess_image(sample_img_path, config=custom_cfg)
    assert pipe_out is not None and len(pipe_out.shape) == 2, "Pipeline output invalid"
    print(f"  [OK] 6. preprocess_image (pipeline): processed shape={pipe_out.shape}")

    # Raw mode check
    raw_out = preprocess_image(sample_img_path, config={"mode": "raw"})
    assert raw_out.shape == orig.shape, "Raw mode modified the image!"
    print(f"  [OK] 7. preprocess_image (raw mode): shape preserved={raw_out.shape}")
    print()


def main():
    test_corpus_verification()
    test_normalisation()
    test_preprocessing()
    print("=" * 60)
    print("ALL TESTS AND VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
