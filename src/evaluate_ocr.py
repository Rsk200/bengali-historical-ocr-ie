#!/usr/bin/env python3
"""
OCR Evaluation and Alignment Module
Computes CER/WER via Levenshtein edit distance and jiwer,
calculates post-OCR correction metrics (CCR, WCR, Change Ratio),
evaluates historical target entity preservation (PERSON, DATE/PERIOD, LOCATION),
and builds full character-level offset mappings with span projection and IoU calculation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import jiwer
import numpy as np
from rapidfuzz.distance import Levenshtein

from src.normalise import (
    normalise_bangla_text,
    strip_bangla_suffixes_for_matching
)


# ==============================================================================
# 1. CER & WER Computation
# ==============================================================================

def compute_cer_wer(
    gold_text: str,
    ocr_text: str,
    normalise: bool = True
) -> Dict[str, Union[float, int]]:
    """
    Compute Character Error Rate (CER) and Word Error Rate (WER) between
    gold standard reference text and OCR hypothesis text.
    """
    g_text = gold_text or ""
    o_text = ocr_text or ""

    if normalise:
        g_text = normalise_bangla_text(g_text, digits_target="bangla")
        o_text = normalise_bangla_text(o_text, digits_target="bangla")

    # Character-level edit distance
    char_dist = int(Levenshtein.distance(g_text, o_text))
    gold_char_len = max(1, len(g_text))
    cer = min(1.0, char_dist / gold_char_len)

    # Word-level edit distance using jiwer
    # Handle empty hypothesis or reference cleanly
    if not g_text.strip() and not o_text.strip():
        wer = 0.0
        word_dist = 0
        gold_word_count = 0
    elif not g_text.strip():
        wer = 1.0
        word_dist = len(o_text.split())
        gold_word_count = 0
    elif not o_text.strip():
        wer = 1.0
        word_dist = len(g_text.split())
        gold_word_count = len(g_text.split())
    else:
        try:
            wer = float(jiwer.wer(g_text, o_text))
        except Exception:
            # Fallback word-level Levenshtein
            g_words = g_text.split()
            o_words = o_text.split()
            word_dist = int(Levenshtein.distance(g_words, o_words))
            wer = word_dist / max(1, len(g_words))
        g_words = g_text.split()
        o_words = o_text.split()
        word_dist = int(Levenshtein.distance(g_words, o_words))
        gold_word_count = len(g_words)

    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "char_distance": char_dist,
        "gold_char_len": len(g_text),
        "word_distance": word_dist,
        "gold_word_count": gold_word_count
    }


def compute_corpus_cer_wer(page_eval_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate per-page CER/WER into Mean, Median, and Micro-averaged metrics.
    """
    if not page_eval_list:
        return {"mean_cer": 0.0, "median_cer": 0.0, "micro_cer": 0.0,
                "mean_wer": 0.0, "median_wer": 0.0, "micro_wer": 0.0}

    cers = [p["cer"] for p in page_eval_list]
    wers = [p["wer"] for p in page_eval_list]

    tot_char_dist = sum(p.get("char_distance", 0) for p in page_eval_list)
    tot_gold_chars = sum(p.get("gold_char_len", 0) for p in page_eval_list)

    tot_word_dist = sum(p.get("word_distance", 0) for p in page_eval_list)
    tot_gold_words = sum(p.get("gold_word_count", 0) for p in page_eval_list)

    micro_cer = tot_char_dist / max(1, tot_gold_chars)
    micro_wer = tot_word_dist / max(1, tot_gold_words)

    return {
        "mean_cer": round(float(np.mean(cers)), 4),
        "median_cer": round(float(np.median(cers)), 4),
        "micro_cer": round(float(micro_cer), 4),
        "mean_wer": round(float(np.mean(wers)), 4),
        "median_wer": round(float(np.median(wers)), 4),
        "micro_wer": round(float(micro_wer), 4)
    }


# ==============================================================================
# 2. Phase 2 Post-OCR Correction Metrics
# ==============================================================================

def compute_correction_metrics(
    raw_ocr_text: str,
    post_ocr_text: str,
    gold_text: str
) -> Dict[str, float]:
    """
    Calculate post-OCR correction metrics:
      - Character Change Rate (CCR) = CER(post_ocr, raw_ocr)
      - Word Change Rate (WCR) = WER(post_ocr, raw_ocr)
      - Change Ratio = CCR / CER(raw_ocr, gold)
      - CER Reduction = raw_cer - post_cer
    """
    raw_vs_post = compute_cer_wer(raw_ocr_text, post_ocr_text)
    ccr = raw_vs_post["cer"]
    wcr = raw_vs_post["wer"]

    raw_eval = compute_cer_wer(gold_text, raw_ocr_text)
    post_eval = compute_cer_wer(gold_text, post_ocr_text)

    raw_cer = raw_eval["cer"]
    post_cer = post_eval["cer"]
    raw_wer = raw_eval["wer"]
    post_wer = post_eval["wer"]

    change_ratio = (ccr / raw_cer) if raw_cer > 0 else 0.0
    cer_reduction = round(raw_cer - post_cer, 4)
    wer_reduction = round(raw_wer - post_wer, 4)

    return {
        "ccr": round(ccr, 4),
        "wcr": round(wcr, 4),
        "raw_cer": round(raw_cer, 4),
        "post_cer": round(post_cer, 4),
        "corrected_cer": round(post_cer, 4),
        "raw_wer": round(raw_wer, 4),
        "post_wer": round(post_wer, 4),
        "corrected_wer": round(post_wer, 4),
        "change_ratio": round(change_ratio, 4),
        "cer_reduction": cer_reduction,
        "wer_reduction": wer_reduction
    }


# ==============================================================================
# 3. Target Entity Accuracy
# ==============================================================================

def compute_target_accuracy(
    gold_entities: List[Dict[str, str]],
    ocr_text: str
) -> Dict[str, Any]:
    """
    Calculate exact-match preservation accuracy of historical target entities:
      - PERSON (Historical persons)
      - DATE/PERIOD (Dates and temporal expressions)
      - LOCATION (Geographic locations)

    Returns recall under:
      1. raw_accuracy: direct substring match in raw OCR text
      2. norm_accuracy: match after Unicode NFC and whitespace normalisation
      3. stemmed_accuracy: match after conservative inflectional suffix stripping
    """
    if not gold_entities:
        return {"overall": {"total": 0, "raw_accuracy": 1.0, "norm_accuracy": 1.0, "stemmed_accuracy": 1.0}, "by_type": {}}

    ocr_norm = normalise_bangla_text(ocr_text, digits_target="bangla")

    type_stats: Dict[str, Dict[str, int]] = {}
    tot_raw_match = 0
    tot_norm_match = 0
    tot_stem_match = 0
    total_entities = len(gold_entities)

    for ent in gold_entities:
        e_type = ent.get("entity_type", ent.get("type", "GENERAL"))
        e_text = ent.get("name", ent.get("text", "")).strip()
        if not e_text:
            continue

        if e_type not in type_stats:
            type_stats[e_type] = {"total": 0, "raw_match": 0, "norm_match": 0, "stem_match": 0}

        type_stats[e_type]["total"] += 1

        # 1. Raw match
        raw_found = e_text in ocr_text
        if raw_found:
            type_stats[e_type]["raw_match"] += 1
            tot_raw_match += 1

        # 2. Normalised match
        e_norm = normalise_bangla_text(e_text, digits_target="bangla")
        norm_found = e_norm in ocr_norm
        if norm_found:
            type_stats[e_type]["norm_match"] += 1
            tot_norm_match += 1

        # 3. Suffix stripped match
        e_stem = strip_bangla_suffixes_for_matching(e_norm)
        stem_found = e_stem in ocr_norm
        if stem_found:
            type_stats[e_type]["stem_match"] += 1
            tot_stem_match += 1

    by_type_results = {}
    for t, s in type_stats.items():
        tot = max(1, s["total"])
        by_type_results[t] = {
            "total": s["total"],
            "raw_accuracy": round(s["raw_match"] / tot, 4),
            "norm_accuracy": round(s["norm_match"] / tot, 4),
            "stemmed_accuracy": round(s["stem_match"] / tot, 4)
        }

    overall_total = max(1, total_entities)
    return {
        "overall": {
            "total": total_entities,
            "raw_accuracy": round(tot_raw_match / overall_total, 4),
            "norm_accuracy": round(tot_norm_match / overall_total, 4),
            "stemmed_accuracy": round(tot_stem_match / overall_total, 4)
        },
        "by_type": by_type_results
    }


# ==============================================================================
# 4. Block-then-Character Alignment and Span IoU
# ==============================================================================

def align_blocks_and_chars(
    gold_text: str,
    ocr_text: str
) -> Dict[str, Any]:
    """
    Performs two-stage alignment:
      Stage 1: Aligns coarse paragraphs/blocks based on double newlines.
      Stage 2: Computes fine character-level offset mappings using Levenshtein opcodes.

    Returns:
      - ocr_to_gold_char_map: Dict[int, Optional[int]]
      - gold_to_ocr_char_map: Dict[int, Optional[int]]
      - block_alignments: List of aligned paragraph block indices
    """
    # 1. Block/Paragraph alignment
    gold_blocks = [b.strip() for b in gold_text.split("\n\n") if b.strip()]
    ocr_blocks = [b.strip() for b in ocr_text.split("\n\n") if b.strip()]

    block_alignments = []
    # Block alignment via Levenshtein distance on block level
    if gold_blocks and ocr_blocks:
        block_ops = Levenshtein.opcodes(ocr_blocks, gold_blocks)
        for tag, s_start, s_end, d_start, d_end in block_ops:
            if tag in ("equal", "replace"):
                for s_i, d_i in zip(range(s_start, s_end), range(d_start, d_end)):
                    block_alignments.append({"ocr_block": s_i, "gold_block": d_i, "tag": tag})

    # 2. Character-level offset mapping across the full text
    opcodes = Levenshtein.opcodes(ocr_text, gold_text)

    ocr_to_gold_char_map: Dict[int, Optional[int]] = {}
    gold_to_ocr_char_map: Dict[int, Optional[int]] = {}

    for tag, src_start, src_end, dest_start, dest_end in opcodes:
        src_len = src_end - src_start
        dest_len = dest_end - dest_start

        if tag == "equal":
            for i in range(src_len):
                ocr_to_gold_char_map[src_start + i] = dest_start + i
                gold_to_ocr_char_map[dest_start + i] = src_start + i

        elif tag == "replace":
            # Map proportional character indices inside substitution chunk
            for i in range(src_len):
                mapped_d = dest_start + int(i * (dest_len / max(1, src_len)))
                mapped_d = min(dest_end - 1, max(dest_start, mapped_d))
                ocr_to_gold_char_map[src_start + i] = mapped_d

            for j in range(dest_len):
                mapped_s = src_start + int(j * (src_len / max(1, dest_len)))
                mapped_s = min(src_end - 1, max(src_start, mapped_s))
                gold_to_ocr_char_map[dest_start + j] = mapped_s

        elif tag == "delete":
            # Present in OCR, deleted from Gold (insertion in OCR)
            for i in range(src_len):
                ocr_to_gold_char_map[src_start + i] = None

        elif tag == "insert":
            # Missing in OCR, present in Gold (deletion in OCR)
            for j in range(dest_len):
                gold_to_ocr_char_map[dest_start + j] = None

    return {
        "ocr_to_gold_char_map": ocr_to_gold_char_map,
        "gold_to_ocr_char_map": gold_to_ocr_char_map,
        "block_alignments": block_alignments
    }


def project_ocr_span_to_gold(
    ocr_start: int,
    ocr_end: int,
    ocr_to_gold_char_map: Dict[int, Optional[int]],
    gold_text_len: int = 1000000
) -> Tuple[int, int]:
    """
    Project an OCR character span (ocr_start, ocr_end) to gold text offsets
    using the character alignment map.
    """
    mapped_positions = []
    for idx in range(ocr_start, ocr_end):
        g_idx = ocr_to_gold_char_map.get(idx)
        if g_idx is not None:
            mapped_positions.append(g_idx)

    if mapped_positions:
        return (min(mapped_positions), max(mapped_positions) + 1)

    # Fallback interpolation if span contains only inserted characters
    # Search backwards for nearest mapped index
    left_g = 0
    for idx in range(ocr_start - 1, -1, -1):
        g_idx = ocr_to_gold_char_map.get(idx)
        if g_idx is not None:
            left_g = g_idx + 1
            break

    # Search forwards for nearest mapped index
    right_g = left_g
    for idx in range(ocr_end, ocr_end + 100):
        g_idx = ocr_to_gold_char_map.get(idx)
        if g_idx is not None:
            right_g = g_idx
            break

    return (min(left_g, gold_text_len), min(max(left_g, right_g), gold_text_len))


def compute_span_iou(
    span_a: Tuple[int, int],
    span_b: Tuple[int, int]
) -> float:
    """
    Compute Character-level Intersection-over-Union (IoU) between two text spans.
    Spans are represented as half-open intervals (start, end).
    """
    s1, e1 = span_a
    s2, e2 = span_b

    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    inter = max(0, inter_end - inter_start)

    union = max(e1, e2) - min(s1, s2)
    if union <= 0:
        return 0.0

    return inter / union
