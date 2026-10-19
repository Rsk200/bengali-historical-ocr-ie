#!/usr/bin/env python3
"""
Step 6 Verification Test Suite:
Real Data Execution on Dev Set (Chapter 1: 11 core prose pages, p001 to p011)
  1. Verification of real Label Studio task files & transcription workspace
  2. Verification of preprocessed Dev scans & test-set guard
  3. Diagnostic check of local OCR execution outputs (Tesseract / Gemini)
  4. Verification of Colab upload package (outputs/colab_dev_package.zip)
"""

import json
from pathlib import Path
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.annotation_utils import prepare_label_studio_tasks, generate_transcription_workspace
from src.preprocess import preprocess_split


def test_task1_label_studio_and_workspace():
    print("\n" + "=" * 65)
    print("TEST 1: LABEL STUDIO TASK FILES & DUAL-ANNOTATOR WORKSPACE")
    print("=" * 65)

    task_dir = Path("data/gold_standard/label_studio_tasks")
    assert task_dir.exists(), "label_studio_tasks directory does not exist!"

    expected_files = {
        "week1_pilot_p001_p005.json": 5,
        "week2_dev_p006_p011.json": 6,
        "week3_test_ch02_p014_p024.json": 11,
        "week4_test_ch08_ch09.json": 16
    }

    total_tasks = 0
    for filename, expected_count in expected_files.items():
        file_path = task_dir / filename
        assert file_path.exists(), f"Missing task file: {filename}"
        with open(file_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        assert len(tasks) == expected_count, f"{filename} expected {expected_count} tasks, got {len(tasks)}"
        total_tasks += len(tasks)
        print(f"  [Pass] {filename}: {len(tasks)} tasks verified.")

    assert total_tasks == 38, f"Expected 38 total core prose tasks, got {total_tasks}"

    # Verify Annotator Workspaces
    for annotator in ["annotator_1", "annotator_2"]:
        a_dir = Path("data/gold_standard") / annotator
        assert a_dir.exists(), f"Missing directory: {a_dir}"
        md_files = list(a_dir.glob("p*.md"))
        assert len(md_files) == 11, f"Expected 11 md files for {annotator}, got {len(md_files)}"

        # Verify specific structural placeholders
        p008_content = (a_dir / "p008.md").read_text(encoding="utf-8")
        assert "| :--- |" in p008_content, f"Table placeholder missing in {annotator}/p008.md"

        p002_content = (a_dir / "p002.md").read_text(encoding="utf-8")
        assert "৮. [" in p002_content, f"8-point list placeholder missing in {annotator}/p002.md"

        p006_content = (a_dir / "p006.md").read_text(encoding="utf-8")
        assert "২১. [" in p006_content, f"21-point list placeholder missing in {annotator}/p006.md"

        p009_content = (a_dir / "p009.md").read_text(encoding="utf-8")
        assert "৬. [" in p009_content, f"6-point list placeholder missing in {annotator}/p009.md"

        print(f"  [Pass] {annotator}: 11 structured markdown templates verified (tables & lists checked).")


def test_task2_preprocessed_dev_scans():
    print("\n" + "=" * 65)
    print("TEST 2: PREPROCESSED DEV SCANS & HELD-OUT TEST GUARD")
    print("=" * 65)

    prep_dir = Path("data/preprocessed_scans/dev")
    assert prep_dir.exists(), "preprocessed_scans/dev directory does not exist!"

    dev_pages = [f"p{i:03d}" for i in range(1, 12)]
    for pid in dev_pages:
        img_p = prep_dir / f"{pid}.png"
        assert img_p.exists(), f"Missing preprocessed image: {img_p}"
        assert img_p.stat().st_size > 50000, f"Image {img_p} is abnormally small!"
    print(f"  [Pass] All 11 preprocessed Dev scans exist and are non-empty.")

    # Verify held-out test split protection
    guard_triggered = False
    try:
        preprocess_split(split="test", allow_test=False)
    except PermissionError:
        guard_triggered = True
    assert guard_triggered, "Test-set read-only protection failed to block preprocessing on 'test' split!"
    print("  [Pass] Held-out test set read-only guard successfully verified on preprocessing.")


def test_task3_ocr_outputs_and_env_example():
    print("\n" + "=" * 65)
    print("TEST 3: LOCAL OCR EXECUTION STATUS & .ENV.EXAMPLE")
    print("=" * 65)

    assert Path(".env.example").exists(), ".env.example does not exist!"
    env_content = Path(".env.example").read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" in env_content, "Missing GEMINI_API_KEY in .env.example"
    assert "TESSERACT_PATH" in env_content, "Missing TESSERACT_PATH in .env.example"
    print("  [Pass] .env.example template verified with all required variable keys.")

    # Check that outputs/ocr has records for p001 to p011
    dev_pages = [f"p{i:03d}" for i in range(1, 12)]
    for sys_name in ["tesseract_raw", "tesseract_preproc", "gemini_raw"]:
        out_sys_dir = Path("outputs/ocr") / sys_name
        assert out_sys_dir.exists(), f"Directory missing: {out_sys_dir}"
        for pid in dev_pages:
            json_file = out_sys_dir / f"{pid}.json"
            assert json_file.exists(), f"Missing {json_file}"
            with open(json_file, "r", encoding="utf-8") as f:
                rec = json.load(f)
            assert rec["page_id"] == pid
            assert rec["chapter_id"] == "ch01"
            assert rec["split"] == "dev"
        print(f"  [Pass] outputs/ocr/{sys_name}: 11 structured JSON records verified.")


def test_task4_colab_package():
    print("\n" + "=" * 65)
    print("TEST 4: GOOGLE COLAB DEV PACKAGE ARCHIVE")
    print("=" * 65)

    zip_p = Path("outputs/colab_dev_package.zip")
    assert zip_p.exists(), "outputs/colab_dev_package.zip does not exist!"
    size_mb = zip_p.stat().st_size / (1024 * 1024)
    assert size_mb > 5.0, f"Zip file is unexpectedly small: {size_mb:.2f} MB"

    with zipfile.ZipFile(zip_p, "r") as zf:
        names = zf.namelist()
        required_items = [
            "notebooks/colab_gpu_runner.ipynb",
            "configs/config.yaml",
            "data/pages.csv",
            "requirements.txt",
            "data/raw_scans/dev/p001.png",
            "data/raw_scans/dev/p013.png",
            "data/preprocessed_scans/dev/p001.png",
            "data/preprocessed_scans/dev/p011.png",
            "src/ocr_wrappers.py",
            "src/preprocess.py"
        ]
        for item in required_items:
            assert item in names, f"Missing required file in zip: {item}"

    print(f"  [Pass] outputs/colab_dev_package.zip verified: {size_mb:.2f} MB ({len(names)} files).")


def main():
    print("=" * 65)
    print("STARTING STEP 6 COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 65)

    test_task1_label_studio_and_workspace()
    test_task2_preprocessed_dev_scans()
    test_task3_ocr_outputs_and_env_example()
    test_task4_colab_package()

    print("\n" + "=" * 65)
    print("ALL STEP 6 VERIFICATION TESTS PASSED SUCCESSFULLY (100%)!")
    print("=" * 65)


if __name__ == "__main__":
    main()
