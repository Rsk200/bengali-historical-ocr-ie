#!/usr/bin/env python3
"""
Phase 2: Post-OCR Correction Ablation Pilot Runner
Executes:
  - Arm B (Contextual Dictionary + Name & Number Guard)
  - Arm C (Gemini LLM Fix Only + Levenshtein Alignment Trimming)
on gemini_raw and tesseract_raw for all 11 Dev pages (p001 - p011).
Computes Character Correction Rate (CCR) and Word Correction Rate (WCR).
Saves to outputs/post_ocr/{source_name}/{arm}/{page_id}.json.
"""

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_ocr import compute_cer_wer
from src.post_ocr_correction import (
    correct_arm_b_contextual_bert_dict,
    correct_arm_c_llm_fix_only
)

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("post_ocr_runner")


def get_gemini_generator():
    load_dotenv(override=True)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured in .env!")

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    candidate_models = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.8-flash"]

    def generate(prompt: str) -> str:
        for model_name in candidate_models:
            for attempt in range(2):
                try:
                    m = genai.GenerativeModel(model_name)
                    res = m.generate_content(
                        prompt,
                        generation_config={"temperature": 0.0}
                    )
                    txt = res.text or ""
                    if txt.strip():
                        return txt
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "Quota" in err_str:
                        logger.warning(f"Model {model_name} quota exceeded. Trying fallback model...")
                        break
                    time.sleep(2)
        return ""

    return generate


def run_post_ocr_pilot(force: bool = False) -> Dict[str, Any]:
    sources = ["gemini_raw", "tesseract_raw"]
    pages = [f"p{i:03d}" for i in range(1, 12)]
    llm_fn = None

    results = {src: {"arm_b": [], "arm_c": []} for src in sources}

    # 1. Run Arm B across sources (fast, offline)
    logger.info(">>> Running Arm B (Dictionary + Name & Number Guard)...")
    for src in sources:
        out_dir = Path(f"outputs/post_ocr/{src}/arm_b")
        out_dir.mkdir(parents=True, exist_ok=True)

        for pid in pages:
            ocr_p = Path(f"outputs/ocr/{src}/{pid}.json")
            if not ocr_p.exists():
                continue
            with open(ocr_p, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)

            raw_txt = ocr_data.get("normalised_text") or ocr_data.get("raw_text") or ""
            out_file = out_dir / f"{pid}.json"

            if out_file.exists() and not force:
                with open(out_file, "r", encoding="utf-8") as f:
                    rec = json.load(f)
            else:
                t0 = time.perf_counter()
                corr_txt = correct_arm_b_contextual_bert_dict(raw_txt)
                dt = round(time.perf_counter() - t0, 3)

                eval_res = compute_cer_wer(raw_txt, corr_txt)
                rec = {
                    "page_id": pid,
                    "source_name": src,
                    "arm": "arm_b",
                    "raw_text": raw_txt,
                    "corrected_text": corr_txt,
                    "ccr": eval_res["cer"],
                    "wcr": eval_res["wer"],
                    "runtime_sec": dt
                }
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)

            results[src]["arm_b"].append(rec)
            logger.info(f"[Arm B] {src}/{pid}: CCR={rec['ccr']:.4f}, WCR={rec['wcr']:.4f}")

    # 2. Run Arm C across sources (Gemini LLM + Trimming)
    logger.info(">>> Running Arm C (Gemini LLM Fix-Only + Alignment Trimming)...")
    if llm_fn is None:
        llm_fn = get_gemini_generator()

    for src in sources:
        out_dir = Path(f"outputs/post_ocr/{src}/arm_c")
        out_dir.mkdir(parents=True, exist_ok=True)

        for pid in pages:
            out_file = out_dir / f"{pid}.json"
            if out_file.exists() and not force:
                with open(out_file, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                if rec.get("ccr", 1.0) < 0.9 and rec.get("corrected_text", "").strip():
                    results[src]["arm_c"].append(rec)
                    logger.info(f"[Arm C (cached)] {src}/{pid}: CCR={rec['ccr']:.4f}, WCR={rec['wcr']:.4f}")
                    continue

            ocr_p = Path(f"outputs/ocr/{src}/{pid}.json")
            if not ocr_p.exists():
                continue
            with open(ocr_p, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)

            raw_txt = ocr_data.get("normalised_text") or ocr_data.get("raw_text") or ""

            t0 = time.perf_counter()
            corr_txt = correct_arm_c_llm_fix_only(raw_txt, llm_generate_fn=llm_fn)
            dt = round(time.perf_counter() - t0, 3)

            eval_res = compute_cer_wer(raw_txt, corr_txt)
            rec = {
                "page_id": pid,
                "source_name": src,
                "arm": "arm_c",
                "raw_text": raw_txt,
                "corrected_text": corr_txt,
                "ccr": eval_res["cer"],
                "wcr": eval_res["wer"],
                "runtime_sec": dt
            }
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)

            results[src]["arm_c"].append(rec)
            logger.info(f"[Arm C] {src}/{pid}: CCR={rec['ccr']:.4f}, WCR={rec['wcr']:.4f} in {dt}s")

            # Respect rate limit
            time.sleep(2)

    return results


if __name__ == "__main__":
    run_post_ocr_pilot(force=False)
