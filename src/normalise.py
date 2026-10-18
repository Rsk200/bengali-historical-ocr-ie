#!/usr/bin/env python3
"""
Bangla Unicode Normalisation Module
Provides robust, switchable text normalisation for OCR evaluation,
lexical matching, and pipeline standardisation.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Union


# Bangla and Western digit translation tables
BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
WESTERN_DIGITS = "0123456789"
WESTERN_TO_BANGLA_TABLE = str.maketrans(WESTERN_DIGITS, BANGLA_DIGITS)
BANGLA_TO_WESTERN_TABLE = str.maketrans(BANGLA_DIGITS, WESTERN_DIGITS)

# Nukta mapping dictionary (decomposed -> precomposed canonical)
NUKTA_DECOMPOSED_MAP = {
    "\u09af\u09bc": "\u09df",  # য + ় -> য় (ya + nukta -> yya)
    "\u09a1\u09bc": "\u09dc",  # ড + ় -> ড় (dda + nukta -> rra)
    "\u09a2\u09bc": "\u09dd",  # ঢ + ় -> ঢ় (ddha + nukta -> rha)
}

# Common inflectional case suffixes in Bangla for entity matching
# Ordered from longest to shortest to avoid partial prefix stripping
INFLECTIONAL_SUFFIXES = [
    # Apostrophe variants: 'র, ’র, ‘র
    r"['’‘`´]র$",
    r"দেরকে$",     # -দেরকে
    r"গুলোতে$",    # -গুলোতে
    r"গুলোর$",     # -গুলোর
    r"গুলোয়$",    # -গুলোয়
    r"গুলো$",      # -গুলো
    r"গুলি$",      # -গুলি
    r"সমূহ$",      # -সমূহ
    r"দের$",       # -দের (plural genitive/objective)
    r"টিতে$",      # -টিতে
    r"টির$",       # -টির
    r"টাতে$",      # -টাতে
    r"টার$",       # -টার
    r"গণ$",        # -গণ
    r"ের$",        # -ের (e-kar + ra genitive, e.g., পাকিস্তানের, রহমানের)
    r"এর$",        # -এর (independent e + ra genitive)
    r"কে$",        # -কে (objective/dative, e.g., ভাষাকে, ঢাকাকে)
    r"তে$",        # -তে (locative, e.g., ভারতে)
    r"য়ে$",       # -য়ে (locative)
    r"য়$",        # -য় (locative, e.g., ঢাকায়, বাংলায়, পূর্ববাংলায়)
    r"র$",         # -র (genitive, e.g., জিন্নাহর, ভাষার)
]


def normalise_unicode_nfc(text: str) -> str:
    """Normalize string using Unicode Canonical Decomposition, followed by Canonical Composition."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def normalise_ya_nukta(text: str) -> str:
    """
    Unify decomposed consonant + nukta into canonical precomposed characters.
    Ensures য় (\\u09df), ড় (\\u09dc), and ঢ় (\\u09dd) are unified to single codepoints.
    Note: Python NFC decomposes these due to Unicode composition exclusions,
    so we normalize to NFC first, then map decomposed sequences to canonical single codepoints.
    """
    if not text:
        return ""

    # First normalize text
    text = unicodedata.normalize("NFC", text)

    # Map decomposed forms to single canonical codepoints
    for decomposed, precomposed in NUKTA_DECOMPOSED_MAP.items():
        text = text.replace(decomposed, precomposed)

    return text


def clean_zwj_zwnj(text: str, mode: str = "standard") -> str:
    """
    Clean redundant or invalid Zero-Width Joiner (ZWJ, \\u200d) and
    Zero-Width Non-Joiner (ZWNJ, \\u200c) while preserving valid ligatures:
      - Valid র্য (ra + hasant + ZWJ + ya / ra + ZWJ + hasant + ya)
      - Valid খণ্ড-ত (\\u09ce or ta + hasant + ZWJ)
      - Valid conjuncts
    """
    if not text:
        return ""

    if mode == "remove_all":
        # Aggressive mode: remove all ZWJ and ZWNJ
        return text.replace("\u200d", "").replace("\u200c", "")

    # 1. Remove duplicate/consecutive ZWJ/ZWNJ
    text = re.sub(r"[\u200c\u200d]{2,}", lambda m: m.group(0)[0], text)

    # 2. Remove ZWJ/ZWNJ at word boundaries or isolated by whitespace/punctuation
    text = re.sub(r"(^|[\s\(\)\[\]\{\}\<\>.,;:!?\"'’‘।॥—–\-\\/])[\u200c\u200d]+", r"\1", text)
    text = re.sub(r"[\u200c\u200d]+([\s\(\)\[\]\{\}\<\>.,;:!?\"'’‘।॥—–\-\\/]|$)", r"\1", text)

    # 3. Remove ZWJ/ZWNJ between non-Indic characters (ASCII, digits, latin)
    text = re.sub(r"([a-zA-Z0-9])[\u200c\u200d]+([a-zA-Z0-9])", r"\1\2", text)

    return text


def normalise_digits(text: str, target: str = "bangla") -> str:
    """
    Convert digits between Western (0-9) and Bangla (০-৯).
    target: "bangla", "western", or "none" / None
    """
    if not text or target in (None, "none", "passthrough"):
        return text

    if target == "bangla":
        return text.translate(WESTERN_TO_BANGLA_TABLE)
    elif target == "western":
        return text.translate(BANGLA_TO_WESTERN_TABLE)
    else:
        raise ValueError(f"Unknown digit target: {target}. Must be 'bangla', 'western', or 'none'.")


def normalise_whitespace_and_punct(text: str) -> str:
    """
    Standardise whitespace and common punctuation:
      - Replace non-breaking spaces (\u00a0, \u202f) with standard space
      - Collapse multiple horizontal spaces to single space
      - Standardise ASCII pipe '|' to Bangla danda '।' when in Bangla context
      - Standardise multiple newlines (max 2)
      - Trim trailing/leading whitespace per line
    """
    if not text:
        return ""

    # Non-breaking space and tabs to space
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\t", " ")

    # Standardise ASCII pipes to Bangla danda:
    # First handle double pipe: '||' -> '।।'
    text = re.sub(r"\|{2,}", "।।", text)
    # Then handle single pipe preceded or followed by text
    text = re.sub(r"(?<=[\u0980-\u09ff\w])\s*\|\s*", "। ", text)
    text = re.sub(r"\s*\|\s*(?=[\u0980-\u09ff\w])", " ।", text)

    # Standardise multiple dandas (e.g., ।।। -> ।।)
    text = re.sub(r"।{3,}", "।।", text)

    # Collapse horizontal spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip line ends
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _count_bangla_characters(s: str) -> int:
    """Count base Bangla characters (excluding signs/hasanta)."""
    return len(re.findall(r"[\u0985-\u09b9\u09ce\u09dc-\u09df]", s))


def strip_bangla_suffixes_for_matching(entity_str: str, min_stem_length: int = 2) -> str:
    """
    Conservatively strip common Bangla inflectional case suffixes
    for fuzzy and exact entity matching evaluation.
    Preserves stem if stripping would make the word shorter than min_stem_length base characters.

    Examples:
      - জিন্নাহ'র -> জিন্নাহ
      - জিন্নাহর -> জিন্নাহ
      - পাকিস্তানের -> পাকিস্তান
      - ঢাকায় -> ঢাকা
      - পূর্ববাংলায় -> পূর্ববাংলা
      - ছাত্রদের -> ছাত্র
    """
    if not entity_str:
        return ""

    cleaned = entity_str.strip().strip(".,;:\"'’‘()[]{}")
    if not cleaned:
        return ""

    # Apply suffix rules from longest to shortest
    for pattern in INFLECTIONAL_SUFFIXES:
        match = re.search(pattern, cleaned)
        if match:
            candidate_stem = cleaned[:match.start()]
            # Ensure base stem is long enough (at least min_stem_length base letters)
            if _count_bangla_characters(candidate_stem) >= min_stem_length:
                cleaned = candidate_stem
                break  # Only strip one inflectional layer

    # Clean any trailing punctuation or whitespace left on the stem
    cleaned = cleaned.rstrip(" '’‘`´-")
    return cleaned


def normalise_bangla_text(
    text: str,
    unicode_nfc: bool = True,
    unify_nukta: bool = True,
    clean_zw: bool = True,
    digits_target: Optional[str] = "bangla",
    standardise_punct: bool = True,
    strip_suffixes: bool = False,
    min_stem_length: int = 2
) -> str:
    """
    Master composite normalisation function applying all enabled pipeline steps.
    """
    if not text:
        return ""

    res = text
    if unicode_nfc:
        res = normalise_unicode_nfc(res)
    if unify_nukta:
        res = normalise_ya_nukta(res)
    if clean_zw:
        res = clean_zwj_zwnj(res, mode="standard")
    if digits_target and digits_target != "none":
        res = normalise_digits(res, target=digits_target)
    if standardise_punct:
        res = normalise_whitespace_and_punct(res)
    if strip_suffixes:
        res = strip_bangla_suffixes_for_matching(res, min_stem_length=min_stem_length)

    return res
