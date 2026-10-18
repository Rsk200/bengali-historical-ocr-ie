#!/usr/bin/env python3
"""
Structure-Aware Chunker and Ablation Strategies (S1, S2, S3)
Supports:
  - S1 (Core Default): Structure-Aware Chunking (<= 2500 chars, chapter > section > paragraph)
  - S2 (Dev Ablation): Fixed-size Chunking (<= 2500 chars, 10% sentence-aligned overlap)
  - S3 (Dev Ablation): Sentence-level Chunking with Preceding Local Context
Maintains exact character offsets (char_start, char_end) relative to the source page text.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union


# Heading patterns for Bengali textbooks
CHAPTER_PATTERN = re.compile(
    r"^(?:#\s+)?(?:প্রথম|দ্বিতীয়|দ্বিতীয়|তৃতীয়|তৃতীয়|চতুর্থ|পঞ্চম|ষষ্ঠ|সপ্তম|অষ্টম|নবম|দশম|একাদশ|দ্বাদশ|অধ্যায়|অধ্যায়|[\d\u09e6-\u09ef]+)\s*(?:অধ্যায়|অধ্যায়)",
    re.IGNORECASE
)
SECTION_PATTERN = re.compile(
    r"^(?:##\s+)?(?:পরিচ্ছেদ|পাঠ)\s*[\d\.\u09e6-\u09ef]+.*",
    re.IGNORECASE
)
EXERCISE_PATTERN = re.compile(
    r"^(?:###\s+)?(?:অনুশীলনী|বহুনির্বাচনি\s*প্রশ্ন|সৃজনশীল\s*প্রশ্ন|নমুনা\s*প্রশ্ন)",
    re.IGNORECASE
)


def split_bangla_sentences(text: str) -> List[Tuple[str, int, int]]:
    """
    Split Bengali text into sentences while preserving exact character offsets.
    Splits on danda (।), ?, !, or double newlines.
    Returns: List of (sentence_text, start_offset, end_offset)
    """
    sentences = []
    # Pattern matches sentence content followed by punctuation or double newline
    pattern = re.compile(r"([^।?!]+(?:[।?!]|\n\n|$))", re.UNICODE)

    for match in pattern.finditer(text):
        s_text = match.group(1)
        if s_text.strip():
            # Get exact character slice
            start = match.start()
            end = match.end()
            sentences.append((s_text, start, end))

    if not sentences and text.strip():
        sentences.append((text, 0, len(text)))

    return sentences


def is_heading_line(line: str) -> Tuple[bool, Optional[str]]:
    """
    Determine if a line represents a heading and classify its level.
    Returns: (is_heading, level) where level is 'chapter', 'section', or 'subheading'.
    """
    s = line.strip()
    if not s:
        return False, None

    # Explicit markdown headings
    if s.startswith("# "):
        return True, "chapter"
    if s.startswith("## "):
        return True, "section"
    if s.startswith("### "):
        return True, "subheading"

    if CHAPTER_PATTERN.match(s):
        return True, "chapter"
    if SECTION_PATTERN.match(s):
        return True, "section"
    if EXERCISE_PATTERN.match(s):
        return True, "section"

    # Short title line heuristics:
    # Length <= 50, does not end in terminal sentence punctuation, not a bullet or list item
    if len(s) <= 50 and not s.endswith(("।", "?", "!", ".", ":", ";", ",", "—", "-")):
        if not re.match(r"^[\d\u09e6-\u09ef]+[\.\)]", s) and not s.startswith(("-", "*", "•", "১.", "২.")):
            return True, "subheading"

    return False, None


# ==============================================================================
# S1: Structure-Aware Chunking (<= 2500 chars)
# ==============================================================================

def chunk_s1_structure_aware(
    page_text: str,
    page_id: str,
    chapter_id: str = "",
    max_chars: int = 2500
) -> List[Dict[str, Any]]:
    """
    Strategy S1 (Core Default):
    Splits text along chapter > section/heading > paragraph boundaries.
    Merges consecutive short paragraphs under the same heading up to max_chars.
    Preserves exact character offsets (char_start, char_end).
    """
    if not page_text.strip():
        return []

    chunks = []
    lines = page_text.split("\n")

    current_heading_path = [chapter_id] if chapter_id else []
    current_chapter = ""
    current_section = ""
    current_subheading = ""

    # Parse structural blocks with their start/end offsets
    blocks = []
    cursor = 0
    for line in lines:
        line_len = len(line)
        line_start = cursor
        line_end = cursor + line_len
        cursor = line_end + 1  # accounts for '\n'

        stripped = line.strip()
        if not stripped:
            continue

        is_head, level = is_heading_line(stripped)
        blocks.append({
            "text": line,
            "stripped": stripped,
            "start": line_start,
            "end": line_end,
            "is_heading": is_head,
            "level": level
        })

    # Group into chunks
    cur_chunk_start: Optional[int] = None
    cur_chunk_end: Optional[int] = None
    cur_chunk_text_parts: List[str] = []
    chunk_index = 1

    def make_heading_path():
        p = []
        if chapter_id:
            p.append(chapter_id)
        if current_chapter:
            p.append(current_chapter)
        if current_section and current_section != current_chapter:
            p.append(current_section)
        if current_subheading and current_subheading not in (current_chapter, current_section):
            p.append(current_subheading)
        return p if p else ["General"]

    def flush_current_chunk():
        nonlocal cur_chunk_start, cur_chunk_end, cur_chunk_text_parts, chunk_index
        if cur_chunk_start is not None and cur_chunk_end is not None and cur_chunk_end > cur_chunk_start:
            raw_slice = page_text[cur_chunk_start:cur_chunk_end]
            if raw_slice.strip():
                cid = f"{page_id}_c{chunk_index:03d}"
                chunks.append({
                    "chunk_id": cid,
                    "page_id": page_id,
                    "chapter_id": chapter_id,
                    "strategy": "S1",
                    "text": raw_slice,
                    "char_start": cur_chunk_start,
                    "char_end": cur_chunk_end,
                    "heading_path": make_heading_path()
                })
                chunk_index += 1
        cur_chunk_start = None
        cur_chunk_end = None
        cur_chunk_text_parts = []

    for block in blocks:
        if block["is_heading"]:
            level = block["level"]
            # A major heading boundary (chapter or section) triggers a chunk flush
            if level in ("chapter", "section"):
                flush_current_chunk()
                if level == "chapter":
                    current_chapter = block["stripped"]
                    current_section = ""
                    current_subheading = ""
                elif level == "section":
                    current_section = block["stripped"]
                    current_subheading = ""
            elif level == "subheading":
                current_subheading = block["stripped"]

        block_len = block["end"] - block["start"]

        # Check if adding this block exceeds max_chars
        if cur_chunk_start is None:
            cur_chunk_start = block["start"]
            cur_chunk_end = block["end"]
            cur_chunk_text_parts.append(block["text"])
        else:
            potential_len = block["end"] - cur_chunk_start
            if potential_len > max_chars:
                flush_current_chunk()
                cur_chunk_start = block["start"]
                cur_chunk_end = block["end"]
                cur_chunk_text_parts.append(block["text"])
            else:
                cur_chunk_end = block["end"]
                cur_chunk_text_parts.append(block["text"])

    flush_current_chunk()
    return chunks


# ==============================================================================
# S2: Fixed Size with 10% Overlap
# ==============================================================================

def chunk_s2_fixed_overlap(
    page_text: str,
    page_id: str,
    chapter_id: str = "",
    max_chars: int = 2500,
    overlap_chars: int = 250
) -> List[Dict[str, Any]]:
    """
    Strategy S2 (Dev Ablation):
    Fixed-size chunks up to max_chars, cut cleanly at sentence boundaries,
    with approximately 10% (overlap_chars) sentence-aligned overlap.
    """
    sentences = split_bangla_sentences(page_text)
    if not sentences:
        return []

    chunks = []
    n = len(sentences)
    chunk_index = 1
    i = 0

    while i < n:
        chunk_start = sentences[i][1]
        chunk_end = sentences[i][2]
        j = i

        while j < n:
            s_text, s_start, s_end = sentences[j]
            if (s_end - chunk_start) <= max_chars:
                chunk_end = s_end
                j += 1
            else:
                # If first sentence itself exceeds max_chars, take at least it
                if j == i:
                    chunk_end = s_end
                    j += 1
                break

        cid = f"{page_id}_c{chunk_index:03d}"
        raw_slice = page_text[chunk_start:chunk_end]
        chunks.append({
            "chunk_id": cid,
            "page_id": page_id,
            "chapter_id": chapter_id,
            "strategy": "S2",
            "text": raw_slice,
            "char_start": chunk_start,
            "char_end": chunk_end,
            "heading_path": [chapter_id] if chapter_id else ["General"]
        })
        chunk_index += 1

        if j >= n:
            break

        # Calculate overlap step-back
        # Step back from j until overlap is approximately overlap_chars
        overlap_accum = 0
        step_back_idx = j
        for k in range(j - 1, i, -1):
            sent_len = sentences[k][2] - sentences[k][1]
            overlap_accum += sent_len
            step_back_idx = k
            if overlap_accum >= overlap_chars:
                break

        # Ensure forward progress
        i = max(step_back_idx, i + 1)

    return chunks


# ==============================================================================
# S3: Sentence-Level with Preceding Context
# ==============================================================================

def chunk_s3_sentence_level(
    page_text: str,
    page_id: str,
    chapter_id: str = "",
    context_sentences_count: int = 2
) -> List[Dict[str, Any]]:
    """
    Strategy S3 (Dev Ablation):
    One chunk per target sentence, supplying immediately preceding sentence(s)
    as local context while explicitly marking target sentence boundaries.
    """
    sentences = split_bangla_sentences(page_text)
    if not sentences:
        return []

    chunks = []
    chunk_index = 1

    for idx, (s_text, s_start, s_end) in enumerate(sentences):
        # Extract preceding context
        ctx_parts = []
        start_ctx = max(0, idx - context_sentences_count)
        for prev_idx in range(start_ctx, idx):
            ctx_parts.append(sentences[prev_idx][0].strip())
        context_str = " ".join(ctx_parts)

        cid = f"{page_id}_c{chunk_index:03d}"
        raw_target = page_text[s_start:s_end]
        chunks.append({
            "chunk_id": cid,
            "page_id": page_id,
            "chapter_id": chapter_id,
            "strategy": "S3",
            "text": raw_target,
            "context_text": context_str,
            "char_start": s_start,
            "char_end": s_end,
            "heading_path": [chapter_id] if chapter_id else ["General"]
        })
        chunk_index += 1

    return chunks


# ==============================================================================
# Master Chunker Dispatcher & Saving
# ==============================================================================

def chunk_page_text(
    page_text: str,
    page_id: str,
    chapter_id: str = "",
    strategy: str = "S1",
    max_chars: int = 2500,
    overlap_chars: int = 250
) -> List[Dict[str, Any]]:
    """
    Master dispatcher for chunking strategies.
    strategy: "S1" (default), "S2", or "S3"
    """
    strat = strategy.upper().strip()
    if strat == "S1":
        return chunk_s1_structure_aware(page_text, page_id, chapter_id, max_chars=max_chars)
    elif strat == "S2":
        return chunk_s2_fixed_overlap(page_text, page_id, chapter_id, max_chars=max_chars, overlap_chars=overlap_chars)
    elif strat == "S3":
        return chunk_s3_sentence_level(page_text, page_id, chapter_id)
    else:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Must be 'S1', 'S2', or 'S3'.")


def save_chunks(
    chunks: List[Dict[str, Any]],
    source_name: str,
    strategy: str,
    page_id: str,
    output_base_dir: Union[str, Path] = "outputs/chunks"
) -> Path:
    """
    Save chunked outputs to outputs/chunks/{source_name}/{strategy}/{page_id}_chunks.json
    """
    out_dir = Path(output_base_dir) / source_name / strategy.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{page_id}_chunks.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return out_file
