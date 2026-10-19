#!/usr/bin/env python3
"""
Master Execution Script for Real OCR on Development Set (p001 to p011)
Executes:
1. tesseract_raw (on data/raw_scans/dev/p001.png - p011.png)
2. tesseract_preproc (on data/preprocessed_scans/dev/p001.png - p011.png)
3. gemini_raw (on data/raw_scans/dev/p001.png - p011.png)
Reports first 300 characters of extracted Bangla text from p001.json.
"""

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv

# Reconfigure stdout for UTF-8 on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("dev_ocr_runner")


def check_tesseract_languages() -> list[str]:
    """Verify tesseract binary and list available languages."""
    load_dotenv()
    tess_cmd = os.environ.get("TESSERACT_PATH", r"C:\Users\Yaad\Tesseract-OCR\tesseract.exe")
    tessdata_prefix = os.environ.get("TESSDATA_PREFIX", r"C:\Users\Yaad\Tesseract-OCR\tessdata")
    os.environ["TESSDATA_PREFIX"] = tessdata_prefix

    try:
        res = subprocess.run(
            [tess_cmd, "--list-langs"],
            capture_output=True,
            text=True,
            check=True
        )
        output = res.stdout + res.stderr
        langs = [line.strip() for line in output.splitlines() if line.strip() and not line.startswith("List")]
        return langs
    except Exception as e:
        logger.error(f"Error checking tesseract languages: {e}")
        return []


def check_gemini_key() -> tuple[bool, str]:
    """Check if GEMINI_API_KEY is configured in environment or .env."""
    load_dotenv(override=True)
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key and key != "your_gemini_api_key_here":
        masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
        return True, masked
    return False, "NOT_CONFIGURED"


def run_batch_ocr(force: bool = True):
    from src.ocr_wrappers import TesseractOCR, GeminiOCR

    csv_path = Path("data/pages.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing pages manifest: {csv_path}")

    df = pd.read_csv(csv_path)
    # Filter 11 core prose pages of dev split
    dev_core_df = df[(df["split"] == "dev") & (df["page_type"] == "core_prose")].copy()
    dev_core_df.sort_values("book_page", inplace=True)
    page_ids = dev_core_df["page_id"].tolist()
    logger.info(f"Dev core prose pages to process ({len(page_ids)} pages): {page_ids}")

    # 1. Initialize OCR systems
    tess_raw = TesseractOCR(system_name="tesseract_raw")
    tess_preproc = TesseractOCR(system_name="tesseract_preproc")
    gemini_raw = GeminiOCR(system_name="gemini_raw")

    results_summary = {
        "tesseract_raw": [],
        "tesseract_preproc": [],
        "gemini_raw": []
    }

    # 2. Run tesseract_raw on data/raw_scans/dev/
    logger.info(">>> Starting tesseract_raw execution on raw scans...")
    for _, row in dev_core_df.iterrows():
        pid = row["page_id"]
        raw_img_path = Path(row["image_path"])
        meta = {
            "book_page": int(row["book_page"]),
            "chapter_id": row["chapter_id"],
            "split": row["split"],
            "page_type": row["page_type"],
            "layout_tags": row["layout_tags"]
        }
        res = tess_raw.run_page(
            page_id=pid,
            image_path=raw_img_path,
            metadata=meta,
            preprocess_config={"mode": "raw"},
            force=force
        )
        results_summary["tesseract_raw"].append({
            "page_id": pid,
            "chars": len(res.get("normalised_text", "")),
            "runtime": res.get("runtime_sec", 0.0)
        })

    # 3. Run tesseract_preproc on data/preprocessed_scans/dev/
    logger.info(">>> Starting tesseract_preproc execution on preprocessed scans...")
    for _, row in dev_core_df.iterrows():
        pid = row["page_id"]
        preproc_img_path = Path(f"data/preprocessed_scans/dev/{pid}.png")
        meta = {
            "book_page": int(row["book_page"]),
            "chapter_id": row["chapter_id"],
            "split": row["split"],
            "page_type": row["page_type"],
            "layout_tags": row["layout_tags"]
        }
        res = tess_preproc.run_page(
            page_id=pid,
            image_path=preproc_img_path,
            metadata=meta,
            preprocess_config={"mode": "raw"},
            force=force
        )
        results_summary["tesseract_preproc"].append({
            "page_id": pid,
            "chars": len(res.get("normalised_text", "")),
            "runtime": res.get("runtime_sec", 0.0)
        })

    # 4. Run gemini_raw on data/raw_scans/dev/
    logger.info(">>> Starting gemini_raw execution on raw scans...")
    for _, row in dev_core_df.iterrows():
        pid = row["page_id"]
        raw_img_path = Path(row["image_path"])
        meta = {
            "book_page": int(row["book_page"]),
            "chapter_id": row["chapter_id"],
            "split": row["split"],
            "page_type": row["page_type"],
            "layout_tags": row["layout_tags"]
        }
        res = gemini_raw.run_page(
            page_id=pid,
            image_path=raw_img_path,
            metadata=meta,
            preprocess_config={"mode": "raw"},
            force=force
        )
        results_summary["gemini_raw"].append({
            "page_id": pid,
            "chars": len(res.get("normalised_text", "")),
            "runtime": res.get("runtime_sec", 0.0)
        })

    return results_summary


def display_p001_preview():
    """Print the first 300 characters of extracted Bangla text from p001.json."""
    systems = ["tesseract_raw", "tesseract_preproc", "gemini_raw"]
    outputs = {}

    print("\n" + "=" * 80)
    print("PAGE p001 REAL EXTRACTED BANGLA TEXT PREVIEW (FIRST 300 CHARACTERS)")
    print("=" * 80)

    for sys_name in systems:
        json_p = Path(f"outputs/ocr/{sys_name}/p001.json")
        print(f"\n--- [System: {sys_name}] ---")
        if not json_p.exists():
            print(f"File not found: {json_p}")
            continue

        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_text = data.get("raw_text", "")
        norm_text = data.get("normalised_text", "")
        runtime = data.get("runtime_sec", 0.0)
        blocks_count = len(data.get("blocks", []))

        print(f"Runtime: {runtime:.2f}s | Raw Chars: {len(raw_text)} | Normalised Chars: {len(norm_text)} | Blocks: {blocks_count}")
        snippet = norm_text[:300].strip()
        if snippet:
            print(snippet)
        else:
            print("(No text extracted - empty or API key not configured)")
        outputs[sys_name] = data

    print("=" * 80 + "\n")
    return outputs


if __name__ == "__main__":
    print("================================================================================")
    print("1. ENVIRONMENT & BINARY VERIFICATION")
    print("================================================================================")
    langs = check_tesseract_languages()
    has_ben = "ben" in langs
    print(f"Tesseract Languages Detected: {langs}")
    print(f"Bengali ('ben') Language Pack Present: {has_ben}")

    has_key, key_info = check_gemini_key()
    print(f"GEMINI_API_KEY Configured in .env: {has_key} (Status: {key_info})")

    print("\n================================================================================")
    print("2. REAL OCR INFERENCE (11 DEV CORE PROSE PAGES: p001 TO p011, force=True)")
    print("================================================================================")
    summary = run_batch_ocr(force=True)

    print("\nSummary of Processed Pages:")
    for sys_name, records in summary.items():
        total_chars = sum(r["chars"] for r in records)
        total_time = sum(r["runtime"] for r in records)
        print(f"  * {sys_name:18s}: {len(records)} pages | {total_chars:,} total chars | {total_time:.2f}s total time")

    print("\n================================================================================")
    print("3. INSPECTION OF p001.json BANGLA TEXT OUTPUT")
    print("================================================================================")
    display_p001_preview()
