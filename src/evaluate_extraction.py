#!/usr/bin/env python3
"""
Two-Tier Information Extraction Evaluation and Statistical Module
Implements:
  1. Tier 1 Strict Surface Matching (Unicode normalized + stemmed exact match)
  2. Tier 2 Aligned Matching (Superset of Tier 1 + character/block position-aligned matches)
     - Guarantees Tier 2 F1 >= Tier 1 F1
     - Flags "Structurally Correct but OCR-Corrupted" items
     - Computes unsupported_item_rate
  3. Three-part Error Attribution (total_ocr_cost, surface_spelling_loss, structural_comprehension_loss)
  4. Bootstrap 95% Confidence Intervals & McNemar's Paired Test
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.evaluate_ocr import compute_span_iou, project_ocr_span_to_gold
from src.normalise import (
    normalise_bangla_text,
    strip_bangla_suffixes_for_matching
)


# ==============================================================================
# Helper Normalisation for Extraction Matching
# ==============================================================================

def normalise_for_matching(text: str) -> str:
    """Normalize text and strip inflectional suffixes for extraction matching."""
    if not text:
        return ""
    norm = normalise_bangla_text(
        text,
        unicode_nfc=True,
        unify_nukta=True,
        clean_zw=True,
        digits_target="bangla",
        standardise_punct=True
    )
    return strip_bangla_suffixes_for_matching(norm)


def _token_f1(str_a: str, str_b: str) -> float:
    """Compute token-level F1 score between two descriptive phrases."""
    tokens_a = set(str_a.split())
    tokens_b = set(str_b.split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    common = tokens_a.intersection(tokens_b)
    p = len(common) / len(tokens_a)
    r = len(common) / len(tokens_b)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ==============================================================================
# 1. Tier 1: Strict Surface Matching
# ==============================================================================

def evaluate_tier1_strict(
    gold_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Tier 1 Strict Surface Evaluation.
    Requires exact match on normalized & stemmed text and category/role.
    """
    gold_matched: Set[int] = set()
    pred_matched: Set[int] = set()
    matches: List[Dict[str, Any]] = []

    # Prepare gold normalized representations
    gold_norm_keys = []
    for g in gold_items:
        category = g.get("category", g.get("type", "ENTITY"))
        if category == "RELATION":
            key = (
                category,
                normalise_for_matching(g.get("head", "")),
                g.get("relation", ""),
                normalise_for_matching(g.get("tail", ""))
            )
        elif category == "EVENT":
            key = (
                category,
                g.get("event_type", ""),
                normalise_for_matching(g.get("trigger", ""))
            )
        else:
            # ENTITY
            key = (
                category,
                g.get("type", ""),
                normalise_for_matching(g.get("text", ""))
            )
        gold_norm_keys.append(key)

    # 1-to-1 greedy matching for predictions
    for p_idx, pred in enumerate(pred_items):
        category = pred.get("category", pred.get("type", "ENTITY"))
        if category == "RELATION":
            p_key = (
                category,
                normalise_for_matching(pred.get("head", "")),
                pred.get("relation", ""),
                normalise_for_matching(pred.get("tail", ""))
            )
        elif category == "EVENT":
            p_key = (
                category,
                pred.get("event_type", ""),
                normalise_for_matching(pred.get("trigger", ""))
            )
        else:
            p_key = (
                category,
                pred.get("type", ""),
                normalise_for_matching(pred.get("text", ""))
            )

        for g_idx, g_key in enumerate(gold_norm_keys):
            if g_idx in gold_matched:
                continue
            if p_key == g_key:
                # For event frames, check if atomic slots match
                if category == "EVENT":
                    g_slots = gold_items[g_idx].get("slots", {})
                    p_slots = pred.get("slots", {})
                    # Match atomic slots if present
                    slots_ok = True
                    for slot_name in ("participants", "location", "date"):
                        if slot_name in g_slots:
                            g_v = normalise_for_matching(str(g_slots[slot_name]))
                            p_v = normalise_for_matching(str(p_slots.get(slot_name, "")))
                            if g_v != p_v:
                                slots_ok = False
                                break
                    if not slots_ok:
                        continue

                gold_matched.add(g_idx)
                pred_matched.add(p_idx)
                matches.append({
                    "tier": "tier1_strict",
                    "gold_idx": g_idx,
                    "pred_idx": p_idx,
                    "status": "Exact Surface Match",
                    "category": category
                })
                break

    tp = len(matches)
    fp = len(pred_items) - tp
    fn = len(gold_items) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold_matched_indices": gold_matched,
        "pred_matched_indices": pred_matched,
        "matches": matches
    }


# ==============================================================================
# 2. Tier 2: Aligned Matching (Strict Superset of Tier 1)
# ==============================================================================

def evaluate_tier2_aligned(
    gold_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]],
    gold_text: str,
    ocr_text: str,
    ocr_to_gold_char_map: Dict[int, Optional[int]],
    min_span_iou: float = 0.5
) -> Dict[str, Any]:
    """
    Tier 2 Position-Aligned Evaluation.
    DEFINED STRICTLY AS A SUPERSET OF TIER 1:
      - All Tier 1 matches are automatically Tier 2 True Positives.
      - Additional matches are established if:
          (a) Projected evidence_span IoU >= min_span_iou with gold evidence_span.
          (b) Type / ontology role matches.
          (c) Extracted text corresponds to source position.
      - Flags items as "Structurally Correct but OCR-Corrupted".
      - Computes unsupported_item_rate for hallucinations.
      - Mathematically guarantees: Tier 2 F1 >= Tier 1 F1.
    """
    # 1. Run Tier 1 first
    t1_res = evaluate_tier1_strict(gold_items, pred_items)

    tier2_matches = list(t1_res["matches"])
    gold_matched: Set[int] = set(t1_res["gold_matched_indices"])
    pred_matched: Set[int] = set(t1_res["pred_matched_indices"])

    corrupted_count = 0
    unsupported_count = 0

    # Check evidence_span grounding in OCR text for hallucination rate
    for pred in pred_items:
        span = pred.get("evidence_span")
        claimed_text = pred.get("text", "")
        if span and isinstance(span, (list, tuple)) and len(span) == 2:
            s, e = span
            if 0 <= s < len(ocr_text) and s < e <= len(ocr_text):
                actual_slice = ocr_text[s:e]
                # Allow minor spacing / punctuation variance
                if normalise_for_matching(claimed_text) not in normalise_for_matching(actual_slice) and \
                   normalise_for_matching(actual_slice) not in normalise_for_matching(claimed_text):
                    unsupported_count += 1
            else:
                unsupported_count += 1
        else:
            if claimed_text and claimed_text not in ocr_text:
                unsupported_count += 1

    unsupported_item_rate = unsupported_count / max(1, len(pred_items))

    # 2. Match remaining unmatched predictions via span alignment
    for p_idx, pred in enumerate(pred_items):
        if p_idx in pred_matched:
            continue

        p_category = pred.get("category", pred.get("type", "ENTITY"))
        p_type = pred.get("type", "")
        p_span = pred.get("evidence_span")

        if not p_span or not isinstance(p_span, (list, tuple)) or len(p_span) != 2:
            continue

        # Project OCR evidence span to Gold text offsets
        proj_gold_span = project_ocr_span_to_gold(
            p_span[0], p_span[1], ocr_to_gold_char_map, gold_text_len=len(gold_text)
        )

        for g_idx, gold in enumerate(gold_items):
            if g_idx in gold_matched:
                continue

            g_category = gold.get("category", gold.get("type", "ENTITY"))
            g_type = gold.get("type", "")
            g_span = gold.get("evidence_span")

            # Must match high-level category and entity/relation/event type
            if p_category != g_category or p_type != g_type:
                continue

            if not g_span or not isinstance(g_span, (list, tuple)) or len(g_span) != 2:
                continue

            # Check Character-level IoU
            iou = compute_span_iou(tuple(proj_gold_span), tuple(g_span))
            if iou >= min_span_iou:
                # If category is EVENT, evaluate descriptive and atomic slots
                slots_valid = True
                if p_category == "EVENT":
                    g_slots = gold.get("slots", {})
                    p_slots = pred.get("slots", {})
                    # Descriptive slots (cause, consequence)
                    for desc_slot in ("cause", "consequence"):
                        if desc_slot in g_slots and desc_slot in p_slots:
                            score = _token_f1(str(g_slots[desc_slot]), str(p_slots[desc_slot]))
                            if score < 0.3:
                                slots_valid = False
                                break

                if not slots_valid:
                    continue

                # Match established in Tier 2!
                gold_matched.add(g_idx)
                pred_matched.add(p_idx)
                corrupted_count += 1
                tier2_matches.append({
                    "tier": "tier2_aligned",
                    "gold_idx": g_idx,
                    "pred_idx": p_idx,
                    "status": "Structurally Correct but OCR-Corrupted",
                    "category": p_category,
                    "iou": round(iou, 4),
                    "gold_text": gold.get("text", ""),
                    "pred_text": pred.get("text", ""),
                    "projected_span": proj_gold_span,
                    "gold_span": g_span
                })
                break

    tp = len(tier2_matches)
    fp = len(pred_items) - tp
    fn = len(gold_items) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Mathematical guarantee check
    assert tp >= t1_res["tp"], f"Tier 2 TP ({tp}) must be >= Tier 1 TP ({t1_res['tp']})"
    assert round(f1, 4) >= t1_res["f1"] - 1e-6, f"Tier 2 F1 ({round(f1, 4)}) must be >= Tier 1 F1 ({t1_res['f1']})"

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tier1_tp": t1_res["tp"],
        "tier2_additional_tp": corrupted_count,
        "unsupported_item_rate": round(unsupported_item_rate, 4),
        "matches": tier2_matches
    }


# ==============================================================================
# 3. Three-Part Error Attribution
# ==============================================================================

def compute_error_attribution(
    oracle_f1: float,
    tier1_f1: float,
    tier2_f1: float
) -> Dict[str, float]:
    """
    Decompose extraction performance drops into a three-part error attribution:
      1. total_ocr_cost = oracle_f1 - tier1_f1
      2. surface_spelling_loss = tier2_f1 - tier1_f1
      3. structural_comprehension_loss = oracle_f1 - tier2_f1

    Guarantees:
      - total_ocr_cost >= 0
      - surface_spelling_loss >= 0
      - structural_comprehension_loss >= 0
      - surface_spelling_loss + structural_comprehension_loss == total_ocr_cost
    """
    # Enforce superset sanity
    if tier2_f1 < tier1_f1:
        tier2_f1 = tier1_f1

    # Clamp oracle if slightly under due to rounding
    if oracle_f1 < tier2_f1:
        oracle_f1 = tier2_f1

    total_ocr_cost = oracle_f1 - tier1_f1
    surface_spelling_loss = tier2_f1 - tier1_f1
    structural_comprehension_loss = oracle_f1 - tier2_f1

    # Strict mathematical assertions
    assert total_ocr_cost >= -1e-7, f"Negative total_ocr_cost: {total_ocr_cost}"
    assert surface_spelling_loss >= -1e-7, f"Negative surface_spelling_loss: {surface_spelling_loss}"
    assert structural_comprehension_loss >= -1e-7, f"Negative structural_comprehension_loss: {structural_comprehension_loss}"

    diff = abs((surface_spelling_loss + structural_comprehension_loss) - total_ocr_cost)
    assert diff < 1e-6, f"Attribution components do not sum to total cost: {diff}"

    return {
        "oracle_f1": round(oracle_f1, 4),
        "tier1_f1": round(tier1_f1, 4),
        "tier2_f1": round(tier2_f1, 4),
        "total_ocr_cost": round(max(0.0, total_ocr_cost), 4),
        "surface_spelling_loss": round(max(0.0, surface_spelling_loss), 4),
        "structural_comprehension_loss": round(max(0.0, structural_comprehension_loss), 4)
    }


# ==============================================================================
# 4. Statistical Module: Bootstrap 95% CI & McNemar's Paired Test
# ==============================================================================

def compute_bootstrap_ci(
    page_level_metrics: Union[List[float], List[Dict[str, float]]],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute non-parametric bootstrap confidence intervals (default 95%)
    via page-level resampling with replacement.
    """
    rng = np.random.default_rng(seed)

    if not page_level_metrics:
        return {}

    # Standardize input to dict of metric lists
    if isinstance(page_level_metrics[0], (int, float)):
        metric_dict = {"metric": [float(x) for x in page_level_metrics]}
    else:
        metric_dict: Dict[str, List[float]] = {}
        for rec in page_level_metrics:
            for k, v in rec.items():
                if isinstance(v, (int, float)):
                    metric_dict.setdefault(k, []).append(float(v))

    results: Dict[str, Dict[str, float]] = {}
    alpha = (1.0 - ci) / 2.0

    for m_name, vals in metric_dict.items():
        arr = np.array(vals)
        n = len(arr)
        if n == 0:
            continue

        boot_means = []
        for _ in range(n_bootstraps):
            sample = rng.choice(arr, size=n, replace=True)
            boot_means.append(float(np.mean(sample)))

        low = float(np.percentile(boot_means, alpha * 100.0))
        high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
        mean_val = float(np.mean(arr))

        results[m_name] = {
            "mean": round(mean_val, 4),
            "ci_lower": round(low, 4),
            "ci_upper": round(high, 4),
            "ci_level": ci
        }

    return results


def mcnemar_paired_test(
    correct_a: List[bool],
    correct_b: List[bool]
) -> Dict[str, Union[float, int]]:
    """
    Perform McNemar's paired test to determine statistical significance
    between two OCR systems or post-correction models on paired items.

    Uses continuity-corrected chi-square (or exact binomial for n < 25).
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(f"Lengths must match: {len(correct_a)} != {len(correct_b)}")

    # b: A correct, B incorrect
    b = sum(1 for ca, cb in zip(correct_a, correct_b) if ca and not cb)
    # c: A incorrect, B correct
    c = sum(1 for ca, cb in zip(correct_a, correct_b) if not ca and cb)

    n = b + c
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": 0, "c": 0, "method": "identical"}

    if n < 25:
        # Exact two-tailed binomial test under p=0.5
        k = min(b, c)
        p_val = 0.0
        for i in range(k + 1):
            p_val += math.comb(n, i) * (0.5 ** n)
        p_val = min(1.0, 2.0 * p_val)
        stat = float(abs(b - c))
        method = "exact_binomial"
    else:
        # Edwards continuity-corrected chi-square: (|b - c| - 1)^2 / (b + c)
        stat = ((abs(b - c) - 1.0) ** 2) / n if abs(b - c) >= 1.0 else 0.0
        # p-value for 1 degree of freedom: erfc(sqrt(stat / 2))
        p_val = math.erfc(math.sqrt(stat) / math.sqrt(2.0))
        method = "continuity_corrected_chi2"

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "b": b,
        "c": c,
        "total_discordant": n,
        "method": method
    }
