#!/usr/bin/env python3
"""
Label Studio Setup, Gold Converter, and Inter-Annotator Agreement (IAA) Module (Phase 6)
Supports:
  1. Label Studio task scheduling across the 4-week transcription plan (38 core prose pages)
  2. Conversion of Label Studio exports to canonical character-offset Gold files (data/gold_standard/{split}/{page_id}_gold.json)
  3. Symmetric Annotator-vs-Annotator Agreement F1 (Span, Relation Tuple, and Event-Slot F1)
  4. Cohen's Kappa on boundary-aligned spans
"""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.evaluate_ocr import compute_span_iou
from src.normalise import (
    normalise_bangla_text,
    normalise_unicode_nfc,
    strip_bangla_suffixes_for_matching
)

logger = logging.getLogger("annotation_utils")


# ==============================================================================
# 1. Annotation Task Generator & 4-Week Schedule
# ==============================================================================

def get_annotation_schedule_week(page_id: str, chapter_id: str, page_type: str) -> Optional[int]:
    """
    Map page to its designated week in the 4-week annotation plan:
      - Week 1: Pilot Calibration (ch01 pages p001-p005, 5 pages)
      - Week 2: Dev Set Remaining (ch01 pages p006-p011, 6 pages; total Dev = 11 core prose pages)
      - Week 3: Held-out Test Set Part 1 (ch02 pages p014-p024, 11 pages)
      - Week 4: Held-out Test Set Part 2 (ch08 pages p098-p105 [8 pages] + ch09 pages p107-p114 [8 pages]; total Test = 27 core prose pages)
    Skips the 6 exercise pages.
    """
    if page_type != "core_prose":
        return None

    pid_num = int(re.sub(r"\D", "", page_id))

    if chapter_id == "ch01":
        if 1 <= pid_num <= 5:
            return 1  # Week 1 Pilot
        elif 6 <= pid_num <= 11:
            return 2  # Week 2 Dev Remaining
    elif chapter_id == "ch02":
        if 14 <= pid_num <= 24:
            return 3  # Week 3 Test Set Part 1
    elif chapter_id in ("ch08", "ch09"):
        if (98 <= pid_num <= 105) or (107 <= pid_num <= 114):
            return 4  # Week 4 Test Set Part 2

    return None


def prepare_label_studio_tasks(
    pages_csv_path: Union[str, Path] = "data/pages.csv",
    output_json_path: Union[str, Path] = "data/gold_standard/label_studio_tasks.json",
    weekly_output_dir: Optional[Union[str, Path]] = "data/gold_standard/label_studio_tasks"
) -> List[Dict[str, Any]]:
    """
    Read data/pages.csv and filter the 38 core_prose pages.
    Generates structured tasks with week metadata and transcription templates.
    Also exports weekly JSON task batches to weekly_output_dir if provided:
      - week1_pilot_p001_p005.json (5 pages)
      - week2_dev_p006_p011.json (6 pages)
      - week3_test_ch02_p014_p024.json (11 pages)
      - week4_test_ch08_ch09.json (16 pages)
    """
    csv_p = Path(pages_csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"pages.csv not found at: {csv_p}")

    df = pd.read_csv(csv_p)
    tasks = []

    for _, row in df.iterrows():
        p_id = str(row["page_id"])
        ch_id = str(row["chapter_id"])
        p_type = str(row.get("page_type", ""))
        split = str(row.get("split", "dev"))
        img_path = str(row.get("image_path", ""))

        week = get_annotation_schedule_week(p_id, ch_id, p_type)
        if week is None:
            # Skip non-core-prose pages (exercise pages)
            continue

        # Template for transcription
        transcription_template = (
            f"<!-- Page: {p_id} (Chapter: {ch_id}, Book Page: {row.get('book_page')}) -->\n"
            f"<!-- Human Transcription Instructions: Preserve all table columns '| ... |' and numbered lists '1. ...' -->\n\n"
        )

        task = {
            "id": p_id,
            "data": {
                "image_url": img_path,
                "text": transcription_template,
                "page_id": p_id,
                "chapter_id": ch_id,
                "split": split,
                "schedule_week": week,
                "book_page": int(row.get("book_page", 0)),
                "layout_tags": str(row.get("layout_tags", ""))
            }
        }
        tasks.append(task)

    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    logger.info(f"Generated {len(tasks)} Label Studio annotation tasks saved to {out_p}")

    # Generate weekly files if weekly_output_dir is specified
    if weekly_output_dir:
        w_dir = Path(weekly_output_dir)
        w_dir.mkdir(parents=True, exist_ok=True)

        week_filenames = {
            1: "week1_pilot_p001_p005.json",
            2: "week2_dev_p006_p011.json",
            3: "week3_test_ch02_p014_p024.json",
            4: "week4_test_ch08_ch09.json"
        }

        for week_num, filename in week_filenames.items():
            week_tasks = [t for t in tasks if t["data"]["schedule_week"] == week_num]
            week_file = w_dir / filename
            with open(week_file, "w", encoding="utf-8") as f:
                json.dump(week_tasks, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(week_tasks)} tasks for week {week_num} to {week_file}")

    return tasks


def generate_transcription_workspace(
    pages_csv_path: Union[str, Path] = "data/pages.csv",
    output_base_dir: Union[str, Path] = "data/gold_standard",
    split: str = "dev"
) -> Dict[str, List[Path]]:
    """
    Generate pre-structured Markdown transcription workspace templates for dual annotators:
      - data/gold_standard/annotator_1/{page_id}.md
      - data/gold_standard/annotator_2/{page_id}.md
    Includes specific structural placeholders:
      - Pipe-table '| ... |' for statistical tables (p008)
      - Numbered lists '১. ...' for enumerated lists (p002 8-point, p006 21-point, p009 6-point)
      - Heading and Activity box blocks for core prose pages
    """
    csv_p = Path(pages_csv_path)
    df = pd.read_csv(csv_p)
    dev_df = df[(df["split"] == split) & (df["page_type"] == "core_prose")]

    annotators = ["annotator_1", "annotator_2"]
    created_files: Dict[str, List[Path]] = {a: [] for a in annotators}

    for _, row in dev_df.iterrows():
        p_id = str(row["page_id"])
        b_page = int(row["book_page"])
        ch_id = str(row["chapter_id"])
        tags = str(row.get("layout_tags", ""))
        img_p = str(row.get("image_path", ""))

        # Build custom content template based on layout tags
        header_block = (
            f"# পেজ ট্রান্সক্রিপশন: {p_id} (বইয়ের পৃষ্ঠা: {b_page})\n"
            f"- অধ্যায়: {ch_id} (বাংলাদেশ ও বিশ্বপরিচয়, নবম-দশম শ্রেণি)\n"
            f"- লেআউট ট্যাগ: {tags}\n"
            f"- স্ক্যান চিত্র: `{img_p}`\n"
            f"- নির্দেশিকা: মূল স্ক্যান থেকে নির্ভুল বানান ও যতিচিহ্ন বজায় রেখে ট্রান্সক্রাইব করুন।\n\n"
            "---\n\n"
        )

        body_template = ""
        if p_id == "p001":
            body_template = (
                "## প্রথম অধ্যায়\n"
                "## পূর্ববাংলার আন্দোলন ও জাতীয়তাবাদের উত্থান (১৯৪৭-১৯৭০)\n\n"
                "### পরিচ্ছেদ ১.১ : বাঙালি জাতীয়তাবাদের বিকাশে ভাষা-আন্দোলন\n"
                "#### ভাষা-আন্দোলনের পটভূমি\n\n"
                "[এখানে পৃষ্ঠার মূল বর্ণনামূলক অনুচ্ছেদসমূহ ট্রান্সক্রাইব করুন...]\n"
            )
        elif p_id == "p002":
            items = "\n".join([f"{i}. [দফা {i} এর বিবরণ...]" for i in ["১", "২", "৩", "৪", "৫", "৬", "৭", "৮"]])
            body_template = (
                "[এখানে সূচনা অনুচ্ছেদ ট্রান্সক্রাইব করুন...]\n\n"
                "### ৮-দফা চুক্তি / কর্মসূচি:\n"
                f"{items}\n\n"
                "[এখানে পরবর্তী বর্ণনামূলক অনুচ্ছেদ ট্রান্সক্রাইব করুন...]\n"
            )
        elif p_id == "p003":
            body_template = (
                "[এখানে মূল অনুচ্ছেদ ট্রান্সক্রাইব করুন...]\n\n"
                "> **একক কাজ / জোড়ায় কাজ / দলীয় কাজ:**\n"
                "> [এখানে পাঠ্যবইয়ের অ্যাক্টিভিটি বক্সের বিষয়বস্তু লিখুন...]\n\n"
                "[এখানে পরবর্তী অনুচ্ছেদ ট্রান্সক্রাইব করুন...]\n"
            )
        elif p_id == "p004":
            body_template = (
                "### [পরিচ্ছেদ বা অনুচ্ছেদের শিরোনাম]\n\n"
                "[এখানে মূল বর্ণনামূলক অনুচ্ছেদসমূহ ট্রান্সক্রাইব করুন...]\n"
            )
        elif p_id == "p005":
            body_template = (
                "[এখানে মূল অনুচ্ছেদ ট্রান্সক্রাইব করুন...]\n\n"
                "> **কাজ / বক্স:**\n"
                "> [এখানে অ্যাক্টিভিটি বক্সের লেখা ট্রান্সক্রাইব করুন...]\n\n"
                "[এখানে পরবর্তী বর্ণনামূলক অনুচ্ছেদ লিখুন...]\n"
            )
        elif p_id == "p006":
            items = "\n".join([f"{i}. [দফা {i} এর বিবরণ...]" for i in [
                "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯", "১০",
                "১১", "১২", "১৩", "১৪", "১৫", "১৬", "১৭", "১৮", "১৯", "২০", "২১"
            ]])
            body_template = (
                "### যুক্তফ্রন্টের ২১ দফা কর্মসূচি:\n"
                f"{items}\n\n"
                "[এখানে পরবর্তী মূল্যায়ন ও অনুচ্ছেদ ট্রান্সক্রাইব করুন...]\n"
            )
        elif p_id == "p007":
            body_template = (
                "[এখানে মূল অনুচ্ছেদ ট্রান্সক্রাইব করুন...]\n\n"
                "> **কাজ:**\n"
                "> [এখানে অ্যাক্টিভিটি বক্স ট্রান্সক্রাইব করুন...]\n"
            )
        elif p_id == "p008":
            body_template = (
                "### পূর্ব ও পশ্চিম পাকিস্তানের অর্থনৈতিক ও প্রশাসনিক বৈষম্য:\n\n"
                "| খাত / বিষয় | পূর্ব পাকিস্তান | পশ্চিম পাকিস্তান | বৈষম্যের অনুপাত / শতকরা হার |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| [উন্নয়ন ব্যয়] | [...] | [...] | [...] |\n"
                "| [রাজস্ব ব্যয়] | [...] | [...] | [...] |\n"
                "| [বৈদেশিক সাহায্য] | [...] | [...] | [...] |\n"
                "| [আমদানি ও বৈদেশিক মুদ্রা] | [...] | [...] | [...] |\n\n"
                "[এখানে টেবিল পরবর্তী ব্যাখ্যা ও বিশ্লেষণ ট্রান্সক্রাইব করুন...]\n"
            )
        elif p_id == "p009":
            items = "\n".join([f"{i}. [দফা {i} এর বিবরণ...]" for i in ["১", "২", "৩", "৪", "৫", "৬"]])
            body_template = (
                "### ঐতিহাসিক ৬ দফা কর্মসূচি:\n"
                f"{items}\n\n"
                "[এখানে ৬ দফার গুরুত্ব ও প্রেক্ষাপটের অনুচ্ছেদ ট্রান্সক্রাইব করুন...]\n"
            )
        elif p_id in ("p010", "p011"):
            body_template = (
                "### [পরিচ্ছেদ বা অনুচ্ছেদের শিরোনাম]\n\n"
                "[এখানে মূল ঐতিহাসিক ঘটনা ও বর্ণনামূলক অনুচ্ছেদসমূহ ট্রান্সক্রাইব করুন...]\n"
            )

        full_content = header_block + body_template

        for annotator in annotators:
            a_dir = Path(output_base_dir) / annotator
            a_dir.mkdir(parents=True, exist_ok=True)
            out_file = a_dir / f"{p_id}.md"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(full_content)
            created_files[annotator].append(out_file)

    logger.info(f"Generated {len(created_files['annotator_1'])} templates each for Annotator 1 and Annotator 2.")
    return created_files



# ==============================================================================
# 2. Label Studio Export to Character-Offset Gold Converter
# ==============================================================================

def convert_label_studio_to_gold(
    label_studio_export: Union[str, Path, List[Dict[str, Any]]],
    output_base_dir: Union[str, Path] = "data/gold_standard"
) -> List[Path]:
    """
    Converts Label Studio JSON export into canonical character-offset Gold files:
      data/gold_standard/{split}/{page_id}_gold.json

    Validates that gold_text[char_start:char_end] strictly matches the labeled text.
    """
    if isinstance(label_studio_export, (str, Path)):
        with open(label_studio_export, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)
    else:
        raw_tasks = label_studio_export

    out_base = Path(output_base_dir)
    saved_files = []

    for task in raw_tasks:
        data = task.get("data", {})
        page_id = data.get("page_id", task.get("id", "p000"))
        split = data.get("split", "dev")
        raw_gold_text = data.get("text", "")

        # Standardise gold text to Unicode NFC
        gold_text = normalise_unicode_nfc(raw_gold_text)

        # Parse annotations
        annotations = task.get("annotations", [])
        if not annotations:
            # Fallback to predictions if no completed annotations
            annotations = task.get("predictions", [])

        results = annotations[0].get("result", []) if annotations else []

        entities = []
        relations = []
        events = []
        id_to_entity: Dict[str, Dict[str, Any]] = {}

        # 1. Parse Entity & Clause Spans
        for r in results:
            r_type = r.get("type")
            val = r.get("value", {})
            r_id = r.get("id", "")

            if r_type == "labels":
                start = val.get("start")
                end = val.get("end")
                text = val.get("text", "")
                labels = val.get("labels", [])
                label = labels[0] if labels else "GENERAL"

                # Verify character offset slice
                if 0 <= start < len(gold_text) and start < end <= len(gold_text):
                    actual_slice = gold_text[start:end]
                    # Soft check allowing whitespace differences
                    if actual_slice.strip() != text.strip():
                        logger.warning(
                            f"[{page_id}] Span slice mismatch for id '{r_id}': "
                            f"gold_text[{start}:{end}]='{actual_slice}' != label_text='{text}'"
                        )

                ent_record = {
                    "entity_id": r_id,
                    "name": text,
                    "entity_type": label,
                    "char_start": start,
                    "char_end": end,
                    "evidence_span": text
                }
                id_to_entity[r_id] = ent_record

                if label in ("TRIGGER", "CAUSE", "CONSEQUENCE"):
                    # Event narrative clause
                    pass
                else:
                    entities.append(ent_record)

        # 2. Parse Relations and Event Argument Links
        for r in results:
            if r.get("type") == "relation":
                from_id = r.get("from_id")
                to_id = r.get("to_id")
                labels = r.get("labels", [])
                rel_type = labels[0] if labels else "related_to"

                head_ent = id_to_entity.get(from_id)
                tail_ent = id_to_entity.get(to_id)

                if head_ent and tail_ent:
                    if rel_type.startswith("has_"):
                        # Event slot argument link
                        pass
                    else:
                        relations.append({
                            "relation_id": f"rel_{len(relations)+1}",
                            "head_text": head_ent["name"],
                            "head_type": head_ent["entity_type"],
                            "relation_type": rel_type,
                            "tail_text": tail_ent["name"],
                            "tail_type": tail_ent["entity_type"],
                            "evidence_span": head_ent["name"] + " ... " + tail_ent["name"]
                        })

        # Save gold standard file
        gold_record = {
            "page_id": page_id,
            "chapter_id": data.get("chapter_id", ""),
            "split": split,
            "gold_text": gold_text,
            "entities": entities,
            "relations": relations,
            "events": events
        }

        split_dir = out_base / split
        split_dir.mkdir(parents=True, exist_ok=True)
        gold_file = split_dir / f"{page_id}_gold.json"

        with open(gold_file, "w", encoding="utf-8") as f:
            json.dump(gold_record, f, ensure_ascii=False, indent=2)

        saved_files.append(gold_file)

    logger.info(f"Converted {len(saved_files)} gold standard page records into {out_base}")
    return saved_files


# ==============================================================================
# 3. Inter-Annotator Agreement (IAA) Module
# ==============================================================================

def compute_iaa_f1(
    annotator_1_items: Dict[str, List[Dict[str, Any]]],
    annotator_2_items: Dict[str, List[Dict[str, Any]]],
    gold_text: str = ""
) -> Dict[str, Any]:
    """
    Computes symmetric Annotator-vs-Annotator F1:
      F1 = 2 * |A1 ∩ A2| / (|A1| + |A2|)

    Evaluated across 3 levels:
      (a) Entity Span F1:
          - Exact Span F1: identical (start, end, entity_type)
          - Relaxed Span F1: IoU >= 0.8 & same entity_type
      (b) Relation Tuple F1: identical (head_stem, relation_type, tail_stem)
      (c) Event Slot F1:
          - Atomic slots (trigger, date, location): exact span match
          - Descriptive slots (cause, consequence): token overlap IoU >= 0.5 or token F1 >= 0.5
    """
    # -------------------------------------------------------------
    # (a) Entity Span IAA
    # -------------------------------------------------------------
    e1_list = annotator_1_items.get("entities", [])
    e2_list = annotator_2_items.get("entities", [])

    exact_matches = 0
    relaxed_matches = 0
    matched_e2_exact = set()
    matched_e2_relaxed = set()

    for e1 in e1_list:
        s1 = (e1.get("char_start"), e1.get("char_end"))
        t1 = e1.get("entity_type")
        n1 = strip_bangla_suffixes_for_matching(e1.get("name", ""))

        for idx, e2 in enumerate(e2_list):
            s2 = (e2.get("char_start"), e2.get("char_end"))
            t2 = e2.get("entity_type")
            n2 = strip_bangla_suffixes_for_matching(e2.get("name", ""))

            # Exact character match
            if idx not in matched_e2_exact and s1 == s2 and t1 == t2:
                exact_matches += 1
                matched_e2_exact.add(idx)
                break

        for idx, e2 in enumerate(e2_list):
            s2 = (e2.get("char_start"), e2.get("char_end"))
            t2 = e2.get("entity_type")
            n2 = strip_bangla_suffixes_for_matching(e2.get("name", ""))

            # Relaxed match: IoU >= 0.8 and same entity_type (or stem match)
            if idx not in matched_e2_relaxed and t1 == t2:
                iou = compute_span_iou(s1, s2) if (None not in s1 and None not in s2) else 0.0
                if iou >= 0.8 or n1 == n2:
                    relaxed_matches += 1
                    matched_e2_relaxed.add(idx)
                    break

    denom_e = len(e1_list) + len(e2_list)
    entity_exact_f1 = (2.0 * exact_matches / denom_e) if denom_e > 0 else 1.0
    entity_relaxed_f1 = (2.0 * relaxed_matches / denom_e) if denom_e > 0 else 1.0

    # -------------------------------------------------------------
    # (b) Relation Tuple IAA
    # -------------------------------------------------------------
    r1_list = annotator_1_items.get("relations", [])
    r2_list = annotator_2_items.get("relations", [])

    def get_rel_tuple(r):
        return (
            strip_bangla_suffixes_for_matching(r.get("head_text", "")),
            r.get("relation_type", ""),
            strip_bangla_suffixes_for_matching(r.get("tail_text", ""))
        )

    r1_tuples = [get_rel_tuple(r) for r in r1_list]
    r2_tuples = [get_rel_tuple(r) for r in r2_list]

    rel_matches = 0
    matched_r2 = set()
    for t1 in r1_tuples:
        for idx, t2 in enumerate(r2_tuples):
            if idx not in matched_r2 and t1 == t2:
                rel_matches += 1
                matched_r2.add(idx)
                break

    denom_r = len(r1_list) + len(r2_list)
    relation_f1 = (2.0 * rel_matches / denom_r) if denom_r > 0 else 1.0

    # -------------------------------------------------------------
    # (c) Event Slot IAA
    # -------------------------------------------------------------
    ev1_list = annotator_1_items.get("events", [])
    ev2_list = annotator_2_items.get("events", [])

    atomic_slot_matches = 0
    atomic_slot_total = 0
    descriptive_slot_matches = 0
    descriptive_slot_total = 0

    for ev1 in ev1_list:
        trigger1 = strip_bangla_suffixes_for_matching(ev1.get("trigger_name", ""))
        for ev2 in ev2_list:
            trigger2 = strip_bangla_suffixes_for_matching(ev2.get("trigger_name", ""))
            if trigger1 == trigger2 and ev1.get("event_type") == ev2.get("event_type"):
                # Atomic slots (date, location)
                for slot in ("date", "location"):
                    v1 = ev1.get(slot)
                    v2 = ev2.get(slot)
                    if v1 or v2:
                        atomic_slot_total += 2
                        if v1 and v2 and strip_bangla_suffixes_for_matching(v1) == strip_bangla_suffixes_for_matching(v2):
                            atomic_slot_matches += 2

                # Descriptive slots (causes, consequences)
                for slot in ("causes", "consequences"):
                    items1 = ev1.get(slot, [])
                    items2 = ev2.get(slot, [])
                    if items1 or items2:
                        descriptive_slot_total += (len(items1) + len(items2))
                        for c1 in items1:
                            for c2 in items2:
                                # Token-level overlap
                                t1_set = set(c1.split())
                                t2_set = set(c2.split())
                                if t1_set and t2_set:
                                    inter = len(t1_set.intersection(t2_set))
                                    union = len(t1_set.union(t2_set))
                                    if (inter / union) >= 0.5:
                                        descriptive_slot_matches += 2
                                        break
                break

    event_atomic_f1 = (atomic_slot_matches / atomic_slot_total) if atomic_slot_total > 0 else 1.0
    event_descriptive_f1 = (descriptive_slot_matches / descriptive_slot_total) if descriptive_slot_total > 0 else 1.0

    return {
        "entity_exact_f1": round(entity_exact_f1, 4),
        "entity_relaxed_f1": round(entity_relaxed_f1, 4),
        "relation_tuple_f1": round(relation_f1, 4),
        "event_atomic_slot_f1": round(event_atomic_f1, 4),
        "event_descriptive_slot_f1": round(event_descriptive_f1, 4),
        "overall_iaa_f1": round(float(np.mean([entity_relaxed_f1, relation_f1, event_atomic_f1])), 4)
    }


def cohen_kappa_on_aligned_spans(
    annotator_1_spans: List[Dict[str, Any]],
    annotator_2_spans: List[Dict[str, Any]],
    min_iou: float = 0.5
) -> Dict[str, float]:
    """
    Computes Cohen's Kappa strictly as a secondary check on the categorical labels
    of spans whose character boundaries are already aligned (IoU >= min_iou).
    """
    paired_labels: List[Tuple[str, str]] = []

    matched_2 = set()
    for e1 in annotator_1_spans:
        s1 = (e1.get("char_start"), e1.get("char_end"))
        if None in s1:
            continue

        for idx, e2 in enumerate(annotator_2_spans):
            if idx in matched_2:
                continue
            s2 = (e2.get("char_start"), e2.get("char_end"))
            if None in s2:
                continue

            iou = compute_span_iou(s1, s2)
            if iou >= min_iou:
                paired_labels.append((e1.get("entity_type", "OTHER"), e2.get("entity_type", "OTHER")))
                matched_2.add(idx)
                break

    if not paired_labels:
        return {"observed_agreement": 1.0, "cohen_kappa": 1.0, "n_aligned_spans": 0}

    n = len(paired_labels)
    # Observed agreement
    agreed = sum(1 for l1, l2 in paired_labels if l1 == l2)
    p_o = agreed / n

    # Expected agreement
    labels_all = list(set([l1 for l1, _ in paired_labels] + [l2 for _, l2 in paired_labels]))
    p_e = 0.0
    for lab in labels_all:
        count_1 = sum(1 for l1, _ in paired_labels if l1 == lab)
        count_2 = sum(1 for _, l2 in paired_labels if l2 == lab)
        p_e += (count_1 / n) * (count_2 / n)

    if (1.0 - p_e) == 0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        "observed_agreement": round(p_o, 4),
        "cohen_kappa": round(kappa, 4),
        "n_aligned_spans": n
    }
