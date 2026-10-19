#!/usr/bin/env python3
"""
Batch Runner for Gemini Raw OCR on Dev Set (p001 to p011)
Processes core prose pages with graceful error handling and rate-limiting.
"""

import json
import logging
import os
from pathlib import Path
import sys
import time
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr_wrappers import GeminiOCR

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gemini_runner")


def run_gemini_dev_pages(force_all: bool = False):
    load_dotenv(override=True)
    ocr = GeminiOCR(system_name="gemini_raw", model_name="gemini-3.8-flash")

    df = pd.read_csv("data/pages.csv")
    dev_core_df = df[(df["split"] == "dev") & (df["page_type"] == "core_prose")].copy()
    dev_core_df.sort_values("book_page", inplace=True)

    results = []
    for _, row in dev_core_df.iterrows():
        pid = row["page_id"]
        out_file = Path(f"outputs/ocr/gemini_raw/{pid}.json")

        # Skip p001 if already populated and not force_all
        if out_file.exists() and not force_all:
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if len(data.get("raw_text", "")) > 100:
                    logger.info(f"[gemini_raw] {pid} already has valid extraction ({len(data['raw_text'])} chars). Skipping.")
                    results.append(data)
                    continue
            except Exception:
                pass

        logger.info(f"[gemini_raw] Transcribing {pid} ({row['book_page']}/11)...")
        meta = {
            "book_page": int(row["book_page"]),
            "chapter_id": row["chapter_id"],
            "split": row["split"],
            "page_type": row["page_type"],
            "layout_tags": row["layout_tags"]
        }
        try:
            res = ocr.run_page(
                page_id=pid,
                image_path=Path(row["image_path"]),
                metadata=meta,
                preprocess_config={"mode": "raw"},
                force=True
            )
            results.append(res)
            logger.info(f"[gemini_raw] Completed {pid}: {len(res.get('raw_text', ''))} chars in {res.get('runtime_sec', 0)}s")
        except Exception as e:
            logger.error(f"[gemini_raw] Failed transcribing {pid}: {e}")

        # Polite delay to respect API rate limits (15 RPM)
        time.sleep(3)

    return results


if __name__ == "__main__":
    run_gemini_dev_pages(force_all=False)
