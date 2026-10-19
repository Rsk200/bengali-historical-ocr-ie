#!/usr/bin/env python3
"""
Phase 3: Real Dev-Set Chunking Script
Executes S1 (Structure-Aware), S2 (Fixed Overlap), and S3 (Sentence-Level)
on gemini_raw, tesseract_raw, and tesseract_preproc OCR outputs for p001 to p011.
Saves to outputs/chunks/{source_name}/{strategy}/{page_id}_chunks.json.
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import chunk_page_text, save_chunks

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("dev_chunker")


def run_dev_chunking() -> Dict[str, Any]:
    sources = ["gemini_raw", "tesseract_raw", "tesseract_preproc"]
    strategies = ["S1", "S2", "S3"]
    pages = [f"p{i:03d}" for i in range(1, 12)]

    stats = {src: {strat: {"count": 0, "total_chars": 0, "min_chars": 999999, "max_chars": 0} for strat in strategies} for src in sources}

    for src in sources:
        for strat in strategies:
            for pid in pages:
                ocr_json_path = Path(f"outputs/ocr/{src}/{pid}.json")
                if not ocr_json_path.exists():
                    logger.warning(f"Missing OCR file: {ocr_json_path}")
                    continue

                with open(ocr_json_path, "r", encoding="utf-8") as f:
                    ocr_data = json.load(f)

                text = ocr_data.get("normalised_text") or ocr_data.get("raw_text") or ""
                chapter_id = ocr_data.get("chapter_id") or "ch01"

                chunks = chunk_page_text(
                    page_text=text,
                    page_id=pid,
                    chapter_id=chapter_id,
                    strategy=strat,
                    max_chars=2500,
                    overlap_chars=250
                )

                # Save output
                out_path = save_chunks(
                    chunks=chunks,
                    source_name=src,
                    strategy=strat,
                    page_id=pid,
                    output_base_dir="outputs/chunks"
                )

                # Collect statistics
                n_chunks = len(chunks)
                stats[src][strat]["count"] += n_chunks
                for c in chunks:
                    c_len = len(c.get("text", ""))
                    stats[src][strat]["total_chars"] += c_len
                    stats[src][strat]["min_chars"] = min(stats[src][strat]["min_chars"], c_len)
                    stats[src][strat]["max_chars"] = max(stats[src][strat]["max_chars"], c_len)

    return stats


if __name__ == "__main__":
    logger.info("Executing Phase 3 Dev-Set Chunking across 11 pages (p001 - p011)...")
    summary = run_dev_chunking()
    logger.info("Dev-Set Chunking Completed.")
