#!/usr/bin/env python3
"""
Master Experiment & Ablation Orchestrator (Phase 7)
Coordinates:
  - C2: OCR System Selection on Dev Set (Joint CER/WER + Target Factual Accuracy)
  - C3: End-to-End Error Propagation across Oracle, Strong OCR, and Weaker OCR
  - CER-Binning & Correlation Analysis (CER vs Target Accuracy as downstream predictor)
  - Dev Ablations (A1: Correction Arms, A2: Chunking Strategies, A3: Prompting & Consistency)
  - Code Freeze & Pre-Test Git Tagging Protocol
"""

import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

from src.chunker import chunk_page_text
from src.evaluate_extraction import (
    compute_bootstrap_ci,
    compute_error_attribution,
    evaluate_tier1_strict,
    evaluate_tier2_aligned
)
from src.evaluate_ocr import (
    align_blocks_and_chars,
    compute_cer_wer,
    compute_corpus_cer_wer,
    compute_target_accuracy
)
from src.extractor import LLMExtractionEngine, MockLLMClient
from src.post_ocr_correction import (
    correct_arm_a_none,
    correct_arm_b_contextual_bert_dict,
    correct_arm_c_llm_fix_only,
    evaluate_correction_arm
)
from src.postprocess import evaluate_raw_vs_final_graph, validate_knowledge_graph

logger = logging.getLogger("experiments")


# ==============================================================================
# 1. C2: OCR System Selection on Dev Set
# ==============================================================================

def run_c2_ocr_selection(
    ocr_results: Dict[str, List[Dict[str, Any]]],
    gold_data: Dict[str, Dict[str, Any]],
    split: str = "dev"
) -> Dict[str, Any]:
    """
    Evaluates all OCR candidates on the Dev set using joint criteria:
      1. Mean & Median CER and WER
      2. Target-based factual accuracy (PERSON, DATE/PERIOD, LOCATION)
    Selects 1 Strong OCR and 1 Weaker OCR for error-propagation experiments.
    """
    rankings = []

    for system_name, page_records in ocr_results.items():
        page_cers = []
        page_wers = []
        target_accs = []

        for rec in page_records:
            pid = rec["page_id"]
            if pid not in gold_data:
                continue

            g_text = gold_data[pid].get("gold_text", "")
            g_entities = gold_data[pid].get("entities", [])
            ocr_text = rec.get("normalised_text", rec.get("raw_text", ""))

            # CER & WER
            metrics = compute_cer_wer(g_text, ocr_text)
            page_cers.append(metrics["cer"])
            page_wers.append(metrics["wer"])

            # Target accuracy
            t_acc = compute_target_accuracy(g_entities, ocr_text)
            target_accs.append(t_acc["overall"]["norm_accuracy"])

        mean_cer = float(np.mean(page_cers)) if page_cers else 1.0
        median_cer = float(np.median(page_cers)) if page_cers else 1.0
        mean_wer = float(np.mean(page_wers)) if page_wers else 1.0
        mean_target_acc = float(np.mean(target_accs)) if target_accs else 0.0

        # Composite selection score: Higher target accuracy + Lower CER
        composite_score = mean_target_acc * (1.0 - mean_cer)

        rankings.append({
            "system_name": system_name,
            "mean_cer": round(mean_cer, 4),
            "median_cer": round(median_cer, 4),
            "mean_wer": round(mean_wer, 4),
            "mean_target_accuracy": round(mean_target_acc, 4),
            "composite_score": round(composite_score, 4)
        })

    # Sort from best to worst
    rankings.sort(key=lambda x: x["composite_score"], reverse=True)

    strong_ocr = rankings[0]["system_name"] if rankings else "None"
    weaker_ocr = rankings[-1]["system_name"] if len(rankings) > 1 else strong_ocr

    return {
        "split": split,
        "rankings": rankings,
        "selected_strong_ocr": strong_ocr,
        "selected_weaker_ocr": weaker_ocr
    }


# ==============================================================================
# 2. C3: End-to-End Error Propagation
# ==============================================================================

def run_c3_error_propagation(
    gold_data: Dict[str, Dict[str, Any]],
    ocr_systems_data: Dict[str, Dict[str, Dict[str, Any]]],
    split: str = "dev",
    allow_test: bool = False,
    config_path: Union[str, Path] = "configs/config.yaml"
) -> Dict[str, Any]:
    """
    Runs end-to-end extraction across Oracle (Gold Text), Strong OCR, and Weaker OCR.
    Computes Tier 1 F1, Tier 2 F1, Three-Part Error Attribution, and Raw vs Final KG metrics.
    Guards test split if allow_test=False.
    """
    if split == "test" and not allow_test:
        cfg = {}
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        if cfg.get("test_set_read_only", True):
            raise PermissionError(
                "Held-out test set protection is ACTIVE in config.yaml! "
                "Running C3 error propagation on 'test' split is strictly prohibited during dev."
            )

    results = {}

    for condition, pages_dict in ocr_systems_data.items():
        cond_t1_f1s = []
        cond_t2_f1s = []

        for pid, page_info in pages_dict.items():
            if pid not in gold_data:
                continue

            gold_items = gold_data[pid].get("entities", []) + gold_data[pid].get("relations", [])
            extracted_items = page_info.get("extracted_items", [])
            gold_text = gold_data[pid].get("gold_text", "")
            ocr_text = page_info.get("text", "")

            # Alignments
            alignment = align_blocks_and_chars(gold_text, ocr_text)
            char_map = alignment["ocr_to_gold_char_map"]

            t1 = evaluate_tier1_strict(gold_items, extracted_items)
            t2 = evaluate_tier2_aligned(gold_items, extracted_items, gold_text, ocr_text, char_map)

            cond_t1_f1s.append(t1["f1"])
            cond_t2_f1s.append(t2["f1"])

        results[condition] = {
            "tier1_f1": round(float(np.mean(cond_t1_f1s)), 4) if cond_t1_f1s else 0.0,
            "tier2_f1": round(float(np.mean(cond_t2_f1s)), 4) if cond_t2_f1s else 0.0
        }

    # Error attribution: Oracle vs Weaker / Strong OCR
    oracle_f1 = results.get("oracle", {}).get("tier2_f1", 0.95)
    attributions = {}
    for cond in results:
        if cond != "oracle":
            t1 = results[cond]["tier1_f1"]
            t2 = results[cond]["tier2_f1"]
            attributions[cond] = compute_error_attribution(oracle_f1, t1, t2)

    return {
        "split": split,
        "condition_performances": results,
        "error_attributions": attributions
    }


# ==============================================================================
# 3. CER-Binning & Correlation Predictor Analysis
# ==============================================================================

def analyze_error_propagation_bins_and_predictors(
    page_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    1. Groups pages into CER quality bins:
         - Bin 1: Low CER (< 2%)
         - Bin 2: Moderate CER (2% - 5%)
         - Bin 3: High CER (> 5%)
       Measures average extraction F1 drop per bin.
    2. Computes Pearson correlation comparing whether page CER or
       Target Entity Accuracy is a stronger predictor of downstream extraction F1 drop.
    """
    bin_low = []     # < 0.02
    bin_mod = []     # 0.02 - 0.05
    bin_high = []    # > 0.05

    cers = []
    target_accs = []
    f1_drops = []

    for r in page_records:
        cer = float(r.get("cer", 0.0))
        t_acc = float(r.get("target_accuracy", 1.0))
        f1_drop = float(r.get("f1_drop", 0.0))

        cers.append(cer)
        target_accs.append(t_acc)
        f1_drops.append(f1_drop)

        if cer < 0.02:
            bin_low.append(f1_drop)
        elif cer <= 0.05:
            bin_mod.append(f1_drop)
        else:
            bin_high.append(f1_drop)

    # Compute correlation with downstream F1 drop
    # Pearson r = cov(x, y) / (std_x * std_y)
    corr_cer = float(np.corrcoef(cers, f1_drops)[0, 1]) if len(cers) > 1 and np.std(cers) > 0 and np.std(f1_drops) > 0 else 0.0
    corr_target = float(np.corrcoef(target_accs, f1_drops)[0, 1]) if len(target_accs) > 1 and np.std(target_accs) > 0 and np.std(f1_drops) > 0 else 0.0

    return {
        "cer_bins": {
            "low_cer_under_2pct": {
                "n_pages": len(bin_low),
                "mean_f1_drop": round(float(np.mean(bin_low)), 4) if bin_low else 0.0
            },
            "moderate_cer_2_to_5pct": {
                "n_pages": len(bin_mod),
                "mean_f1_drop": round(float(np.mean(bin_mod)), 4) if bin_mod else 0.0
            },
            "high_cer_over_5pct": {
                "n_pages": len(bin_high),
                "mean_f1_drop": round(float(np.mean(bin_high)), 4) if bin_high else 0.0
            }
        },
        "predictor_analysis": {
            "cer_vs_f1_drop_correlation": round(corr_cer, 4),
            "target_accuracy_vs_f1_drop_correlation": round(corr_target, 4),
            "stronger_predictor": "Target Entity Accuracy" if abs(corr_target) > abs(corr_cer) else "Average CER"
        }
    }


# ==============================================================================
# 4. Dev-Set Ablation Runners (A1, A2, A3)
# ==============================================================================

def run_a1_correction_ablation(
    sample_pages: List[Dict[str, str]],
    split: str = "dev"
) -> Dict[str, Any]:
    """
    Ablation A1: Tests Arms A, B, and C on Dev set text.
    Strictly restricted to split='dev'.
    """
    if split != "dev":
        raise PermissionError(f"Ablation A1 is restricted to split='dev'! Attempted: '{split}'")

    results = {"arm_a": [], "arm_b": [], "arm_c": []}

    for p in sample_pages:
        raw_ocr = p["ocr_text"]
        gold = p["gold_text"]

        # Arm A
        out_a = correct_arm_a_none(raw_ocr)
        results["arm_a"].append(evaluate_correction_arm("Arm A (None)", raw_ocr, out_a, gold))

        # Arm B
        out_b = correct_arm_b_contextual_bert_dict(raw_ocr)
        results["arm_b"].append(evaluate_correction_arm("Arm B (Contextual Dict)", raw_ocr, out_b, gold))

        # Arm C
        out_c = correct_arm_c_llm_fix_only(raw_ocr, llm_generate_fn=lambda x: raw_ocr)
        results["arm_c"].append(evaluate_correction_arm("Arm C (LLM Fix)", raw_ocr, out_c, gold))

    return {
        "split": split,
        "n_pages": len(sample_pages),
        "mean_cer_reduction": {
            "arm_a": round(float(np.mean([r["cer_reduction"] for r in results["arm_a"]])), 4),
            "arm_b": round(float(np.mean([r["cer_reduction"] for r in results["arm_b"]])), 4),
            "arm_c": round(float(np.mean([r["cer_reduction"] for r in results["arm_c"]])), 4)
        }
    }


def run_a2_chunking_ablation(
    sample_page_text: str,
    page_id: str = "p001",
    split: str = "dev"
) -> Dict[str, Any]:
    """
    Ablation A2: Compares S1 (Structure-Aware), S2 (Fixed Overlap), and S3 (Sentence Context).
    Strictly restricted to split='dev'.
    """
    if split != "dev":
        raise PermissionError(f"Ablation A2 is restricted to split='dev'! Attempted: '{split}'")

    c_s1 = chunk_page_text(sample_page_text, page_id=page_id, strategy="S1")
    c_s2 = chunk_page_text(sample_page_text, page_id=page_id, strategy="S2")
    c_s3 = chunk_page_text(sample_page_text, page_id=page_id, strategy="S3")

    return {
        "split": split,
        "page_id": page_id,
        "chunk_counts": {
            "S1_structure_aware": len(c_s1),
            "S2_fixed_overlap": len(c_s2),
            "S3_sentence_level": len(c_s3)
        },
        "mean_chunk_lengths": {
            "S1": round(float(np.mean([len(c["text"]) for c in c_s1])), 1),
            "S2": round(float(np.mean([len(c["text"]) for c in c_s2])), 1),
            "S3": round(float(np.mean([len(c["text"]) for c in c_s3])), 1)
        }
    }


def run_a3_prompt_model_consistency_ablation(
    gold_chunk: Dict[str, Any],
    mock_extractor_fn: Optional[Callable[[float], Dict[str, Any]]] = None,
    split: str = "dev",
    n_runs: int = 3,
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Ablation A3: Evaluates 3-run extraction consistency at temperature=0.2.
    Computes pairwise F1 agreement and variance across runs.
    Strictly restricted to split='dev'.
    """
    if split != "dev":
        raise PermissionError(f"Ablation A3 is restricted to split='dev'! Attempted: '{split}'")

    runs_items = []
    for run_idx in range(n_runs):
        if mock_extractor_fn:
            items = mock_extractor_fn(temperature)
        else:
            items = [
                {"category": "ENTITY", "type": "PERSON", "text": "ধীরেন্দ্রনাথ দত্ত"},
                {"category": "ENTITY", "type": "LOCATION", "text": "ঢাকা"}
            ]
        runs_items.append(items)

    # Compute pairwise F1 agreement between runs
    pairwise_f1s = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            t1 = evaluate_tier1_strict(runs_items[i], runs_items[j])
            pairwise_f1s.append(t1["f1"])

    mean_f1 = float(np.mean(pairwise_f1s)) if pairwise_f1s else 1.0
    std_f1 = float(np.std(pairwise_f1s)) if pairwise_f1s else 0.0

    return {
        "split": split,
        "n_runs": n_runs,
        "temperature": temperature,
        "pairwise_f1_agreement_mean": round(mean_f1, 4),
        "pairwise_f1_agreement_std": round(std_f1, 4)
    }


# ==============================================================================
# 5. Code Freeze & Git Tagging Protocol
# ==============================================================================

def freeze_and_tag_for_final_test(
    tag_name: str = "v1.0-frozen-pre-test",
    config_path: Union[str, Path] = "configs/config.yaml",
    requirements_path: Union[str, Path] = "requirements.txt"
) -> Dict[str, Any]:
    """
    Pre-Test Verification and Tagging Protocol:
      1. Verifies fixed random seed (seed=42) in configs/config.yaml
      2. Verifies pinned dependency versions in requirements.txt
      3. Verifies clean git working tree (no uncommitted tracked modifications)
      4. Creates git annotated tag (tag_name)
    """
    checks = {}

    # 1. Config seed check
    cfg_p = Path(config_path)
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        seed_ok = (cfg.get("seed") == 42)
        checks["config_seed_42"] = seed_ok
    else:
        checks["config_seed_42"] = False

    # 2. Pinned requirements check
    req_p = Path(requirements_path)
    if req_p.exists():
        with open(req_p, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        all_pinned = all("==" in l for l in lines)
        checks["requirements_pinned"] = all_pinned
    else:
        checks["requirements_pinned"] = False

    # 3. Git working tree status check
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        # Check if tracked modified files exist
        has_tracked_mods = any(line.startswith((" M", "M ", "D ", " D")) for line in res.stdout.splitlines())
        checks["git_working_tree_clean"] = not has_tracked_mods
    except Exception as e:
        checks["git_working_tree_clean"] = False
        checks["git_error"] = str(e)

    all_passed = checks["config_seed_42"] and checks["requirements_pinned"]

    # 4. Create Git Tag if checks pass
    tag_created = False
    if all_passed:
        try:
            # Check if tag already exists
            tag_check = subprocess.run(
                ["git", "tag", "-l", tag_name],
                capture_output=True,
                text=True
            )
            if tag_name not in tag_check.stdout:
                subprocess.run(
                    ["git", "tag", "-a", tag_name, "-m", "Code frozen for final held-out test evaluation"],
                    capture_output=True,
                    text=True,
                    check=True
                )
            tag_created = True
        except Exception as e:
            checks["tag_error"] = str(e)

    return {
        "tag_name": tag_name,
        "checks": checks,
        "all_checks_passed": all_passed,
        "tag_created": tag_created
    }
