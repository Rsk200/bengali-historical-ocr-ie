#!/usr/bin/env python3
"""
Phase 0 Corpus Extraction Script
Extracts 44 textbook pages from NCTB Class 9-10 BGS PDF at 300 DPI,
generates data/pages.csv with layout tags, and locks test set images as read-only.
"""

import argparse
import logging
import os
from pathlib import Path
import stat
import sys
import fitz  # PyMuPDF
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("logs/phase0_extraction.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("phase0")


SPECIAL_LAYOUT_TAGS = {
    "p002": "prose,enumerated_list_8_point",
    "p003": "prose,activity_box",
    "p005": "prose,activity_box",
    "p006": "prose,enumerated_list_21_point",
    "p007": "prose,activity_box",
    "p008": "prose,statistical_table",
    "p009": "prose,enumerated_list_6_point",
}


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_layout_tags(page_id: str, page_type: str) -> str:
    if page_id in SPECIAL_LAYOUT_TAGS:
        return SPECIAL_LAYOUT_TAGS[page_id]
    if page_type == "core_prose":
        return "prose,heading"
    if page_type == "exercise":
        return "mcq,creative_questions"
    return "prose"


def build_corpus_manifest(config: dict) -> list[dict]:
    """Build metadata entries for all 44 pages based on config definitions."""
    offset = config.get("pdf_page_offset", 6)
    manifest = []

    chapters = config.get("chapters", {})
    for ch_id, ch_info in chapters.items():
        split = ch_info["split"]
        for r in ch_info["ranges"]:
            start_bp = r["start_book_page"]
            end_bp = r["end_book_page"]
            p_type = r["page_type"]

            for bp in range(start_bp, end_bp + 1):
                pid = f"p{bp:03d}"
                pdf_p = bp + offset
                img_rel_path = f"data/raw_scans/{split}/{pid}.png"
                manifest.append({
                    "page_id": pid,
                    "book_page": bp,
                    "pdf_page": pdf_p,
                    "chapter_id": ch_id,
                    "split": split,
                    "page_type": p_type,
                    "layout_tags": get_layout_tags(pid, p_type),
                    "image_path": img_rel_path
                })
    return manifest


def set_read_only(file_path: Path):
    """Set file to read-only mode."""
    os.chmod(file_path, stat.S_IREAD)


def remove_read_only(file_path: Path):
    """Remove read-only mode if file exists and is write-protected."""
    if file_path.exists():
        os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)


def extract_pages(config_path: Path, force: bool = False):
    config = load_config(config_path)
    dpi = config.get("dpi", 300)
    pdf_filename = config.get("pdf_filename")
    pdf_path = Path(pdf_filename)

    if not pdf_path.exists():
        # Check current directory
        alt_path = Path(__file__).resolve().parent.parent / pdf_filename
        if alt_path.exists():
            pdf_path = alt_path
        else:
            raise FileNotFoundError(f"Source PDF file not found at: {pdf_path}")

    logger.info(f"Opening source PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    logger.info(f"Total pages in PDF: {total_pages}")

    expected_total_pages = 206
    if total_pages != expected_total_pages:
        logger.warning(f"Expected {expected_total_pages} pages, but found {total_pages} pages!")
    else:
        logger.info(f"PDF page count verification successful: {total_pages} pages.")

    manifest = build_corpus_manifest(config)
    logger.info(f"Manifest created with {len(manifest)} pages across 4 chapters.")

    # 300 DPI zoom matrix: standard PDF resolution is 72 points per inch
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    rendered_count = 0
    skipped_count = 0

    for item in manifest:
        out_path = Path(item["image_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not force:
            logger.info(f"Skipping existing page: {out_path} (resumable mode)")
            skipped_count += 1
            continue

        if out_path.exists() and force:
            remove_read_only(out_path)

        # PyMuPDF is 0-indexed, pdf_page is 1-indexed
        pdf_page_idx = item["pdf_page"] - 1
        page = doc.load_page(pdf_page_idx)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out_path))
        rendered_count += 1
        logger.info(f"Rendered [{item['page_id']}] (Book P.{item['book_page']}, PDF P.{item['pdf_page']}) -> {out_path} ({pix.width}x{pix.height} @ {dpi} DPI)")

    doc.close()
    logger.info(f"Extraction summary: {rendered_count} rendered, {skipped_count} skipped.")

    # Generate data/pages.csv
    csv_path = Path("data/pages.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(manifest)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved manifest to {csv_path} with {len(df)} rows.")

    # Apply read-only mode to test set if configured
    if config.get("test_set_read_only", True):
        logger.info("Locking test set images in data/raw_scans/test/ to Read-Only mode...")
        test_dir = Path("data/raw_scans/test")
        test_images = list(test_dir.glob("*.png"))
        for test_img in test_images:
            set_read_only(test_img)
        logger.info(f"Locked {len(test_images)} test set images as read-only.")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Corpus Extraction")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Force re-rendering even if image files exist")
    args = parser.parse_args()

    extract_pages(Path(args.config), force=args.force)


if __name__ == "__main__":
    main()
