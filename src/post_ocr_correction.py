#!/usr/bin/env python3
"""
Post-OCR Correction Ablation Module (Phase 2, Dev-set RQ3)
Implements:
  - Arm A (None): Raw uncorrected OCR text (pipeline default)
  - Arm B (Contextual BERT + Dictionary): Lightweight spell-checker with
    NER & Rule-based Name & Number Guard (prevents corruption of entities/dates)
  - Arm C (LLM Fix Only): Zero-shot paragraph prompt with
    Character-Level Alignment Trimming (trims hallucinated additions & protects digits)
  - Evaluation wrapper computing CER, WER, CCR, WCR, and Change Ratio
"""

import json
import logging
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from rapidfuzz.distance import Levenshtein

from src.evaluate_ocr import compute_cer_wer, compute_correction_metrics
from src.normalise import normalise_bangla_text

logger = logging.getLogger("post_ocr_correction")

# Protected historical entities and lexical tokens that should never be altered
PROTECTED_HISTORICAL_NAMES: Set[str] = {
    "ধীরেন্দ্রনাথ দত্ত", "লিয়াকত আলী খান", "খাজা নাজিমুদ্দীন", "মোহাম্মদ আলী জিন্নাহ",
    "শেখ মুজিবুর রহমান", "মাওলানা ভাসানী", "রফিক উদ্দিন", "আবুল বরকত", "সালাম", "জব্বার",
    "ঢাকা", "করাচি", "পূর্ববাংলা", "পশ্চিম পাকিস্তান", "পাকিস্তান গণপরিষদ", "যুক্তফ্রন্ট"
}

CORE_BANGLA_DICTIONARY: Set[str] = {
    "আন্দোলন", "জাতীয়তাবাদ", "পূর্ববাংলা", "ভাষা", "রাষ্ট্রভাষা", "পরিষদ", "গণপরিষদ",
    "অধিবেশন", "ঐতিহাসিক", "দাবি", "প্রধানমন্ত্রী", "ঘোষণা", "প্রতিবাদ", "ধর্মঘট",
    "ফেব্রুয়ারি", "ফেব্রুয়ারি", "মার্চ", "জানুয়ারি", "জানুয়ারি", "আগস্ট", "ডিসেম্বর",
    "শহীদ", "মিছিল", "ছাত্র", "শাসন", "ব্রিটিশ", "পাকিস্তান", "ভারত", "স্বাধীনতা", "সংগ্রাম", "যুক্তফ্রন্ট"
}

COMMON_INFLECTION_SUFFIXES: Tuple[str, ...] = (
    "ের", "এর", "দের", "দেরকে", "কে", "তে", "য়ে", "য়", "ে", "র", "গুলো", "গুলি", "সমূহ", "টিতে", "টির", "টাতে", "টার"
)


# ==============================================================================
# Arm A: None (Baseline / Pipeline Default)
# ==============================================================================

def correct_arm_a_none(ocr_text: str) -> str:
    """Arm A: Returns raw uncorrected OCR text directly."""
    return ocr_text


# ==============================================================================
# Arm B: Contextual Spell-Checker with Name & Number Guard
# ==============================================================================

def is_protected_token(word: str) -> bool:
    """
    Check if a word/token should be protected from spell-correction:
      - Contains Bengali or Western digits/dates (e.g., ১৯৫২, ২১শে)
      - Matches protected historical person/place names or stems
      - Single-character tokens or punctuation
    """
    w = word.strip(".,;:!?\"'’‘()[]{}")
    if not w:
        return True

    # Digit / year check
    if re.search(r"[\d\u09e6-\u09ef]", w):
        return True

    # Name check
    for p_name in PROTECTED_HISTORICAL_NAMES:
        if w in p_name or p_name in w:
            return True

    return False


def correct_arm_b_contextual_bert_dict(
    ocr_text: str,
    dictionary: Optional[Set[str]] = None,
    bert_scorer: Optional[Callable[[str, int], str]] = None
) -> str:
    """
    Arm B: Contextual spell-checker with explicit Name & Number Guard.
    1. Detects and locks historical names and dates/numbers.
    2. Identifies out-of-vocabulary words.
    3. Finds candidate corrections within Levenshtein distance <= 2.
    4. Selects candidate using BERT masked token scoring (or dictionary frequency fallback).
    """
    vocab = dictionary or CORE_BANGLA_DICTIONARY
    words = ocr_text.split()
    corrected_words = []

    for w_idx, word in enumerate(words):
        stripped = word.strip(".,;:!?\"'’‘()[]{}")
        prefix = word[:word.find(stripped)] if stripped in word else ""
        suffix = word[word.find(stripped) + len(stripped):] if stripped in word else ""

        # SAFEGUARD: Name & Number Guard
        if is_protected_token(stripped):
            corrected_words.append(word)
            continue

        # If word is already in vocabulary or an inflected form of a vocabulary word, keep it
        is_inflected_vocab = any(
            stripped.startswith(cand) and stripped[len(cand):] in COMMON_INFLECTION_SUFFIXES
            for cand in vocab
        )
        if stripped in vocab or is_inflected_vocab:
            corrected_words.append(word)
            continue

        # Find closest candidate in vocabulary (edit distance <= 2)
        best_cand = stripped
        best_dist = 3

        # 1. Direct word match
        for cand in vocab:
            d = Levenshtein.distance(stripped, cand)
            if d < best_dist and d <= 2:
                best_dist = d
                best_cand = cand

        # 2. Inflected stem match (e.g., 'ঐতহাসিকের' -> 'ঐতিহাসিকের')
        for sfx in COMMON_INFLECTION_SUFFIXES:
            if stripped.endswith(sfx) and len(stripped) > len(sfx):
                stem_cand = stripped[:-len(sfx)]
                for cand in vocab:
                    d = Levenshtein.distance(stem_cand, cand)
                    if d < best_dist and d <= 2:
                        best_dist = d
                        best_cand = cand + sfx

        # If BERT scorer provided (e.g. on Colab GPU), use it for disambiguation
        if bert_scorer and best_dist <= 2:
            try:
                best_cand = bert_scorer(ocr_text, w_idx)
            except Exception:
                pass

        corrected_words.append(f"{prefix}{best_cand}{suffix}")

    return " ".join(corrected_words)


# ==============================================================================
# Arm C: LLM Fix Only with Alignment Trimming
# ==============================================================================

def trim_overgeneration_by_alignment(
    ocr_text: str,
    corrected_text: str,
    max_length_ratio: float = 1.15
) -> str:
    """
    Safeguard for Arm C:
    1. Uses Levenshtein opcodes to align LLM output against source OCR text.
    2. Trims conversational introductions, explanations, or hallucinated clauses
       exceeding max_length_ratio.
    3. Restores any original dates and numbers that were silently altered.
    """
    clean_corr = corrected_text.strip()

    # 1. Strip conversational introductory prefixes
    prefix_patterns = [
        r"^সংশোধিত\s*পাঠ(?:\s*:)?\s*",
        r"^সংশোধিত\s*অনুচ্ছেদ(?:\s*:)?\s*",
        r"^এখানে\s*সংশোধিত.*?(?:হলো|হল)(?:\s*:)?\s*",
        r"^Here is the corrected text:\s*"
    ]
    for pat in prefix_patterns:
        clean_corr = re.sub(pat, "", clean_corr, flags=re.IGNORECASE).strip()

    # 2. Check length ratio constraint
    if len(clean_corr) > (len(ocr_text) * max_length_ratio) and len(ocr_text) > 0:
        logger.warning(
            f"Arm C output length ({len(clean_corr)}) exceeds {max_length_ratio}x OCR length ({len(ocr_text)}). "
            f"Applying alignment trimming..."
        )
        # Find longest common alignment window and cut off large hallucinated additions
        ops = list(Levenshtein.opcodes(clean_corr, ocr_text))
        first_match = 0
        cutoff = len(clean_corr)

        for op in ops:
            if op.tag == "equal":
                first_match = op.src_start
                break

        for op in ops:
            # If a large addition exists in clean_corr (delete in src relative to ocr_text)
            # when ocr_text is already near its end
            if op.tag == "delete" and (op.dest_start >= len(ocr_text) - 5 or (op.src_end - op.src_start) >= 20):
                cutoff = op.src_start
                break

        clean_corr = clean_corr[first_match:cutoff].strip()
        if ocr_text.endswith("।") and not clean_corr.endswith("।"):
            clean_corr += "।"

    # 3. Safeguard: Restore original 4-digit historical years and numbers from OCR
    ocr_numbers = re.findall(r"\b[\d\u09e6-\u09ef]{2,4}\b", ocr_text)
    corr_numbers = re.findall(r"\b[\d\u09e6-\u09ef]{2,4}\b", clean_corr)

    if ocr_numbers != corr_numbers and len(ocr_numbers) == len(corr_numbers):
        # Restore original OCR numbers positionally
        for old_n, new_n in zip(ocr_numbers, corr_numbers):
            if old_n != new_n:
                clean_corr = clean_corr.replace(new_n, old_n, 1)

    return clean_corr


def correct_arm_c_llm_fix_only(
    ocr_text: str,
    llm_generate_fn: Optional[Callable[[str], str]] = None
) -> str:
    """
    Arm C: Paragraph-level zero-shot prompt instructing the LLM:
    'Correct only OCR-induced spelling errors in the following Bangla paragraph;
    do not add any new information, do not alter historical dates or numbers, and do not rewrite sentences.'
    Applies alignment trimming and number preservation.
    """
    if not llm_generate_fn:
        # Offline / passthrough fallback
        return ocr_text

    prompt = (
        "Correct only OCR-induced spelling errors in the following Bangla paragraph; "
        "do not add any new information, do not alter historical dates or numbers, and do not rewrite sentences.\n\n"
        f"TEXT:\n\"\"\"\n{ocr_text}\n\"\"\"\n\n"
        "Return ONLY the corrected text without any introduction, markdown, or commentary."
    )

    raw_llm_output = llm_generate_fn(prompt)
    trimmed_output = trim_overgeneration_by_alignment(ocr_text, raw_llm_output)
    return trimmed_output


# ==============================================================================
# Evaluation Wrapper
# ==============================================================================

def evaluate_correction_arm(
    arm_name: str,
    raw_ocr_text: str,
    corrected_text: str,
    gold_text: str
) -> Dict[str, Any]:
    """
    Evaluates a correction arm by computing pre- vs post-correction metrics
    using src/evaluate_ocr.py.
    """
    corr_metrics = compute_correction_metrics(
        raw_ocr_text=raw_ocr_text,
        post_ocr_text=corrected_text,
        gold_text=gold_text
    )

    eval_raw = compute_cer_wer(gold_text, raw_ocr_text)
    eval_post = compute_cer_wer(gold_text, corrected_text)

    return {
        "arm_name": arm_name,
        "raw_cer": eval_raw["cer"],
        "post_cer": eval_post["cer"],
        "cer_reduction": round(eval_raw["cer"] - eval_post["cer"], 4),
        "raw_wer": eval_raw["wer"],
        "post_wer": eval_post["wer"],
        "ccr": corr_metrics["ccr"],
        "wcr": corr_metrics["wcr"],
        "change_ratio": corr_metrics["change_ratio"]
    }
