#!/usr/bin/env python3
"""
Synthetic Noise Generator for Bengali OCR
Injects realistic, controlled OCR errors into clean Bengali text:
  - Nukta loss (ড় -> ড, ঢ় -> ঢ, য় -> য)
  - Vowel sign (kar) drop/confusion (ি, ী, ু, ূ, ে, ৈ, ো, ৌ, া)
  - Yuktakshar splitting / consonant cluster corruption (ন্দ -> নদ, ন্দ্র -> ন্দ, ক্ত -> কত)
  - Visually similar character substitution (ব <-> র, ত <-> ভ, ক <-> ফ, খ <-> থ)
  - Digit misrecognition (৪ <-> ৮, ১ <-> ৭, ০ <-> ২)
"""

import random
from typing import Optional


# Visual character confusion mappings in Bengali OCR
CHAR_CONFUSIONS = {
    "ব": ["র", "ক"],
    "র": ["ব", "ন"],
    "ত": ["ভ", "ও"],
    "ভ": ["ত", "ড"],
    "ড": ["ড়", "ভ"],
    "ড়": ["ড", "র"],
    "ঢ়": ["ঢ"],
    "ঢ": ["ঢ়"],
    "ক": ["ফ", "ব"],
    "ফ": ["ক"],
    "খ": ["থ"],
    "থ": ["খ"],
    "য": ["য়", "ঘ"],
    "য়": ["য"],
    "ন": ["ল", "র"],
    "ল": ["ন"],
    "ম": ["স"],
    "স": ["ম"],
    "ণ": ["ন"],
}

# Bengali digit confusions
DIGIT_CONFUSIONS = {
    "৪": ["৮"],
    "৮": ["৪"],
    "১": ["৭"],
    "৭": ["১"],
    "০": ["২"],
    "২": ["০"],
    "৩": ["৫"],
    "৫": ["৩"],
}

# Common yuktakshar corruptions in Bengali OCR
YUKTAKSHAR_SPLITS = {
    "ন্দ্র": "ন্দ",      # e.g., ধীরেন্দ্রনাথ -> ধীরেন্দনাথ
    "ন্দ": "নদ",
    "ন্ত": "নত",
    "ক্ত": "কত",
    "ঙ্গ": "নগ",
    "ন্ধ": "নধ",
    "ণ্ড": "নড",
    "স্থ": "সথ",
    "ম্প": "মপ",
    "স্ট": "সট",
    "ঙ্ক": "নক",
    "ক্ষ": "খ",
    "জ্ঞ": "গ",
    "ষ্ণ": "ষন",
}

# Nukta removals
NUKTA_DROPS = {
    "ড়": "ড",
    "ঢ়": "ঢ",
    "য়": "য",
    "\u09bc": "",  # isolated nukta drop
}

# Vowel kar drops
VOWEL_KARS = ["ি", "ী", "ু", "ূ", "ে", "ৈ", "ো", "ৌ", "া"]


def inject_synthetic_noise(
    text: str,
    target_cer: float = 0.05,
    seed: Optional[int] = 42
) -> str:
    """
    Inject realistic Bengali OCR noise targeting an approximate Character Error Rate (CER).

    Noise types applied probabilistically:
      - Yuktakshar splitting (~30% of noise events)
      - Visually similar character confusion (~30% of noise events)
      - Vowel sign (kar) dropping (~20% of noise events)
      - Nukta removal (~15% of noise events)
      - Digit confusion (~5% of noise events)
    """
    if not text or target_cer <= 0.0:
        return text

    if seed is not None:
        random.seed(seed)

    # 1. First pass: yuktakshar substitution
    modified_text = text
    for cluster, corrupted in YUKTAKSHAR_SPLITS.items():
        if cluster in modified_text and random.random() < (target_cer * 4.0):
            # Replace at least one occurrence
            modified_text = modified_text.replace(cluster, corrupted, 1)

    # 2. Second pass: character-by-character stochastic degradation
    chars = list(modified_text)
    n = len(chars)

    for i in range(n):
        if random.random() >= target_cer:
            continue

        c = chars[i]

        # Choose error mode
        dice = random.random()

        if dice < 0.30 and c in CHAR_CONFUSIONS:
            chars[i] = random.choice(CHAR_CONFUSIONS[c])

        elif dice < 0.50 and c in NUKTA_DROPS:
            chars[i] = NUKTA_DROPS[c]

        elif dice < 0.70 and c in VOWEL_KARS:
            # Drop the kar
            chars[i] = ""

        elif dice < 0.85 and c in DIGIT_CONFUSIONS:
            chars[i] = random.choice(DIGIT_CONFUSIONS[c])

        elif dice < 0.95 and c == " ":
            # Occasional space deletion or double space
            chars[i] = "  " if random.random() < 0.5 else ""

        else:
            # Minor random deletion or swap
            if i + 1 < n and chars[i + 1] != " ":
                chars[i], chars[i + 1] = chars[i + 1], chars[i]

    return "".join(chars)
