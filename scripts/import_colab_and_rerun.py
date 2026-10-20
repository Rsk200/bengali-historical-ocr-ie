#!/usr/bin/env python3
"""
Colab Export Importer and Master Benchmark Re-runner (Phase 1, 2 & 5).
Automatically unzips and integrates colab_export.zip (containing EasyOCR, Surya,
Marker, Qwen2-VL, BanglaBERT Arm B, and Open-Weight Qwen2.5-7B) into project directories,
re-runs the master benchmark suite across C2, C3, and A1-A3, and synchronizes the
regenerated Dev Benchmark Report.
"""

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any, Dict, List, Optional

# Ensure root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.run_dev_master_benchmark import (
    execute_a1_correction_ablation,
    execute_a2_chunking_ablation,
    execute_a3_prompt_consistency_ablation,
    execute_c2_ocr_selection,
    execute_c3_error_propagation,
    generate_dev_benchmark_report,
    load_dev_gold
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("colab_importer")


def find_colab_export(search_dirs: List[Path]) -> Optional[Path]:
    """Finds colab_export.zip or colab_export directory in candidate locations."""
    for d in search_dirs:
        if not d.exists():
            continue
        # Check zip file
        zip_candidate = d / "colab_export.zip" if d.is_dir() else d
        if zip_candidate.is_file() and zip_candidate.name.endswith(".zip"):
            return zip_candidate
        # Check directory
        dir_candidate = d / "colab_export"
        if dir_candidate.is_dir():
            return dir_candidate
    return None


def copy_matching_json(src_dir: Path, dest_dir: Path, prefix: str = "p") -> int:
    """Copies all matching p*.json files from src_dir to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_dir.glob(f"{prefix}*.json"):
        shutil.copy2(f, dest_dir / f.name)
        count += 1
    return count


def import_colab_artifacts(export_path: Path) -> Dict[str, Any]:
    """Unzips if necessary and copies artifacts to their destination directories."""
    imported_summary = {"ocr": {}, "post_ocr": {}, "correction": {}, "extractions": {}}

    if export_path.is_file() and export_path.suffix.lower() == ".zip":
        logger.info(f"Extracting {export_path.name} directly into {root_dir} ...")
        with zipfile.ZipFile(export_path, "r") as zf:
            zf.extractall(root_dir)
        logger.info("Extraction complete!")

    # Check imported directories
    for mod in ["easyocr", "marker_ocr", "surya", "qwen_vl"]:
        p = root_dir / "outputs" / "ocr" / mod
        if p.exists():
            imported_summary["ocr"][mod] = len(list(p.glob("*.json")))

    for src in ["gemini_raw", "tesseract_raw"]:
        p_post = root_dir / "outputs" / "post_ocr" / src / "arm_b"
        if p_post.exists():
            imported_summary["post_ocr"].setdefault(src, {})["arm_b"] = len(list(p_post.glob("*.json")))
        p_corr = root_dir / "outputs" / "correction" / "arm_b" / src
        if p_corr.exists():
            imported_summary["correction"].setdefault(src, {})["arm_b"] = len(list(p_corr.glob("*.json")))
        p_ext = root_dir / "outputs" / "extractions" / src / "qwen2_5_7b"
        if p_ext.exists():
            imported_summary["extractions"].setdefault(src, {})["qwen2_5_7b"] = len(list(p_ext.glob("*.json")))

    return imported_summary


def sync_report_to_artifact():
    """Synchronizes regenerated dev_benchmark_report.md to artifact directory."""
    src_rep = root_dir / "outputs" / "results" / "dev" / "dev_benchmark_report.md"
    art_rep = Path("C:/Users/Yaad/.gemini/antigravity/brain/741db29d-032f-4cf6-88f8-879ce460bd3a/dev_benchmark_report.md")
    if src_rep.exists() and art_rep.parent.exists():
        try:
            shutil.copy2(src_rep, art_rep)
            logger.info(f"Synchronized Dev Benchmark Report to artifact: {art_rep}")
        except Exception as e:
            logger.warning(f"Could not sync artifact report: {e}")


def main():
    parser = argparse.ArgumentParser(description="Import Colab outputs and re-run master benchmark.")
    parser.add_argument("--zip_path", type=str, default=None, help="Explicit path to colab_export.zip")
    parser.add_argument("--rerun-only", action="store_true", help="Re-run benchmark without importing new zip")
    args = parser.parse_args()

    search_dirs = [
        root_dir / "outputs" / "colab_export.zip",
        root_dir / "colab_export.zip",
        Path.home() / "Downloads" / "colab_export.zip",
        root_dir / "outputs",
        root_dir,
        Path.home() / "Downloads"
    ]

    target_export = Path(args.zip_path) if args.zip_path else find_colab_export(search_dirs)

    if not args.rerun_only:
        if not target_export or not target_export.exists():
            logger.warning(
                f"No colab_export.zip found in search paths:\n" +
                "\n".join([f"  - {d}" for d in search_dirs]) +
                "\nPlease place 'colab_export.zip' in 'outputs/' or provide '--zip_path <path>'."
            )
            print("\nTip: To re-run the benchmark with currently available local outputs, run:")
            print("  python scripts/import_colab_and_rerun.py --rerun-only\n")
            sys.exit(1)

        logger.info(f"Found Colab export package: {target_export}")
        import_summary = import_colab_artifacts(target_export)
        logger.info("Import summary: " + json.dumps(import_summary, indent=2))
    else:
        logger.info("Skipping unpack (--rerun-only specified). Re-running benchmark with current outputs.")

    # Re-running benchmark
    logger.info("Triggering Master Development Benchmark suite on available outputs...")
    gold_data = load_dev_gold()

    iaa_path = root_dir / "outputs" / "results" / "dev" / "iaa_summary.json"
    if iaa_path.exists():
        with open(iaa_path, "r", encoding="utf-8") as f:
            iaa_res = json.load(f)
    else:
        logger.error(f"IAA summary not found at {iaa_path}")
        sys.exit(1)

    c2_res = execute_c2_ocr_selection(gold_data)
    c3_res = execute_c3_error_propagation(gold_data)
    a1_res = execute_a1_correction_ablation(gold_data, c3_res["condition_performances"])
    a2_res = execute_a2_chunking_ablation(gold_data, c3_res["condition_performances"])
    a3_res = execute_a3_prompt_consistency_ablation(gold_data)

    generate_dev_benchmark_report(iaa_res, c2_res, c3_res, a1_res, a2_res, a3_res)
    sync_report_to_artifact()

    print("\n" + "=" * 70)
    print("COLAB IMPORT & MASTER DEV BENCHMARK RE-RUN COMPLETE!")
    print(f"Strong OCR: {c2_res['selected_strong_ocr']} | Weaker OCR: {c2_res['selected_weaker_ocr']}")
    print("Updated Report: outputs/results/dev/dev_benchmark_report.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
