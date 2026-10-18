#!/usr/bin/env python3
"""
Post-Processing & Knowledge Graph Construction Module (Phase 5b)
Implements:
  1. Digit, Date, and Suffix Normalisation with Non-Destructive Matching Keys
  2. Logged Entity Resolution (alias containment, stem equality, audit log)
  3. Overlap De-duplication and Conservative Event Merging
  4. Graph Validation (self-loops, causal cycles, chronological date consistency)
  5. Dual Evaluation Helper (Raw Output vs Final Graph Comparison)
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.evaluate_extraction import evaluate_tier1_strict, evaluate_tier2_aligned
from src.normalise import (
    normalise_bangla_text,
    normalise_digits,
    strip_bangla_suffixes_for_matching
)
from src.schema import (
    ChunkExtractionOutput,
    EntityType,
    ExtractedEntity,
    ExtractedEventFrame,
    ExtractedRelation,
    RelationType
)

logger = logging.getLogger("postprocess")

BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
WESTERN_DIGITS = "0123456789"
TO_WESTERN_TRANS = str.maketrans(BANGLA_DIGITS, WESTERN_DIGITS)
TO_BANGLA_TRANS = str.maketrans(WESTERN_DIGITS, BANGLA_DIGITS)


# ==============================================================================
# 1. Digit, Date & Suffix Normalisation
# ==============================================================================

def extract_year_from_text(text: str) -> Optional[int]:
    """Extract a 4-digit Gregorian year (e.g., 1948, 1952, 1971) from Bengali/Western text."""
    if not text:
        return None
    w_text = text.translate(TO_WESTERN_TRANS)
    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", w_text)
    return int(m.group(1)) if m else None


def compute_match_key(name: str) -> str:
    """
    Compute non-destructive normalized matching key for entity resolution:
    Applies Unicode NFC, digit standardisation, and conservative suffix stripping.
    """
    if not name:
        return ""
    norm = normalise_bangla_text(
        name,
        unicode_nfc=True,
        unify_nukta=True,
        clean_zw=True,
        digits_target="bangla",
        standardise_punct=True
    )
    return strip_bangla_suffixes_for_matching(norm)


# ==============================================================================
# 2. Logged Entity Resolution
# ==============================================================================

def resolve_entities(
    entities: List[Dict[str, Any]],
    page_id: str = "",
    log_file_path: Union[str, Path] = "logs/entity_merges.json"
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    """
    Resolve and merge entity aliases within the same entity_type.
    Uses:
      - Match-key equality (stem equality after suffix stripping)
      - Substring / alias containment (e.g., 'শেখ মুজিব' in 'শেখ মুজিবুর রহমান')
    Logs every merge decision to logs/entity_merges.json.

    Returns:
      - canonical_entities: deduplicated list of entity records
      - alias_to_canonical: map from any alias surface form to canonical name
      - merge_logs: list of logged merge records
    """
    if not entities:
        return [], {}, []

    # Group by entity_type
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entities:
        t = e.get("entity_type", "GENERAL")
        by_type[t].append(e)

    alias_to_canonical: Dict[str, str] = {}
    merge_logs: List[Dict[str, Any]] = []
    canonical_entities: List[Dict[str, Any]] = []

    def canonical_score(ent: Dict[str, Any]) -> Tuple[bool, int]:
        name = ent.get("name", "").strip()
        stem = strip_bangla_suffixes_for_matching(name)
        # 1. Prefer unsuffixed base forms (stem == name) over suffixed inflections
        is_unsuffixed = (stem == name)
        # 2. Prefer longer base stem (full names like 'শেখ মুজিবুর রহমান' over 'শেখ মুজিব')
        return (is_unsuffixed, len(stem))

    for e_type, group in by_type.items():
        # Compute match keys and canonical candidates
        clusters: List[Dict[str, Any]] = []

        # Sort so that unsuffixed base names and fuller canonical titles are processed first
        sorted_group = sorted(group, key=canonical_score, reverse=True)

        for ent in sorted_group:
            name = ent.get("name", "").strip()
            if not name:
                continue

            m_key = compute_match_key(name)
            matched_cluster = None

            for cluster in clusters:
                c_canonical = cluster["canonical_name"]
                c_key = cluster["match_key"]

                # Case A: Exact match key equality (handles inflections like 'ধীরেন্দ্রনাথ দত্তের' == 'ধীরেন্দ্রনাথ দত্ত')
                if m_key == c_key:
                    matched_cluster = cluster
                    reason = "Stem match after suffix stripping"
                    break

                # Case B: Alias containment within PERSON (e.g. 'শেখ মুজিব' in 'শেখ মুজিবুর রহমান')
                if e_type in (EntityType.PERSON.value, "PERSON"):
                    if (m_key in c_key and len(m_key) >= 4) or (c_key in m_key and len(c_key) >= 4):
                        matched_cluster = cluster
                        reason = "Alias containment under PERSON type"
                        break

            if matched_cluster:
                # Merge into existing cluster
                matched_cluster["aliases"].add(name)
                alias_to_canonical[name] = matched_cluster["canonical_name"]

                log_entry = {
                    "source_alias": name,
                    "canonical_name": matched_cluster["canonical_name"],
                    "entity_type": e_type,
                    "reason": reason,
                    "page_id": page_id
                }
                merge_logs.append(log_entry)
            else:
                # Form new cluster with canonical base name
                new_cluster = {
                    "canonical_name": name,
                    "match_key": m_key,
                    "entity_type": e_type,
                    "aliases": {name},
                    "representative_entity": ent
                }
                clusters.append(new_cluster)
                alias_to_canonical[name] = name

        for c in clusters:
            rep = dict(c["representative_entity"])
            rep["name"] = c["canonical_name"]
            rep["aliases"] = list(c["aliases"])
            canonical_entities.append(rep)

    # Save merge logs
    log_p = Path(log_file_path)
    log_p.parent.mkdir(parents=True, exist_ok=True)
    existing_logs = []
    if log_p.exists():
        try:
            with open(log_p, "r", encoding="utf-8") as f:
                existing_logs = json.load(f)
        except Exception:
            existing_logs = []

    combined_logs = existing_logs + merge_logs
    with open(log_p, "w", encoding="utf-8") as f:
        json.dump(combined_logs, f, ensure_ascii=False, indent=2)

    return canonical_entities, alias_to_canonical, merge_logs


# ==============================================================================
# 3. Overlap De-duplication & Conservative Event Merging
# ==============================================================================

def deduplicate_relations(
    relations: List[Dict[str, Any]],
    alias_to_canonical: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    De-duplicate relation triples after updating heads/tails to canonical entity names.
    """
    seen_triples = set()
    deduped = []

    for r in relations:
        raw_head = r.get("head_text", "")
        raw_tail = r.get("tail_text", "")
        rel_type = r.get("relation_type", "")

        c_head = alias_to_canonical.get(raw_head, raw_head)
        c_tail = alias_to_canonical.get(raw_tail, raw_tail)

        triple_key = (c_head, rel_type, c_tail)
        if triple_key not in seen_triples:
            seen_triples.add(triple_key)
            updated_r = dict(r)
            updated_r["head_text"] = c_head
            updated_r["tail_text"] = c_tail
            deduped.append(updated_r)

    return deduped


def merge_events_conservatively(
    events: List[Dict[str, Any]],
    alias_to_canonical: Dict[str, str],
    page_id: str = "",
    log_file_path: Union[str, Path] = "logs/event_merges.json"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Conservative Event Merging:
    Merges two event frames ONLY when:
      (a) They share the same event_type, AND
      (b) Their normalized evidence_spans point to the same textual occurrence
          (high token/char overlap or identical trigger in same sentence).
    Merges participants, causes, and consequences without losing distinct occurrences.
    """
    if not events:
        return [], []

    merged_events: List[Dict[str, Any]] = []
    event_logs: List[Dict[str, Any]] = []

    for ev in events:
        ev_type = ev.get("event_type", "EVENT")
        trigger = compute_match_key(ev.get("trigger_name", ""))
        span = ev.get("evidence_span", "").strip()
        span_norm = normalise_bangla_text(span, digits_target="bangla")

        matched_event = None
        for existing in merged_events:
            ex_type = existing.get("event_type", "EVENT")
            ex_trigger = compute_match_key(existing.get("trigger_name", ""))
            ex_span_norm = normalise_bangla_text(existing.get("evidence_span", ""), digits_target="bangla")

            if ev_type == ex_type and trigger == ex_trigger:
                # Check evidence span overlap
                if span_norm in ex_span_norm or ex_span_norm in span_norm:
                    matched_event = existing
                    break

        if matched_event:
            # Merge participants
            existing_part_names = {p["name"] for p in matched_event.get("participants", [])}
            for p in ev.get("participants", []):
                p_copy = dict(p)
                p_copy["name"] = alias_to_canonical.get(p_copy["name"], p_copy["name"])
                if p_copy["name"] not in existing_part_names:
                    matched_event["participants"].append(p_copy)
                    existing_part_names.add(p_copy["name"])

            # Merge causes & consequences
            for c in ev.get("causes", []):
                if c not in matched_event.get("causes", []):
                    matched_event["causes"].append(c)
            for cq in ev.get("consequences", []):
                if cq not in matched_event.get("consequences", []):
                    matched_event["consequences"].append(cq)

            log_entry = {
                "merged_event_id": ev.get("event_id"),
                "into_event_id": matched_event.get("event_id"),
                "event_type": ev_type,
                "trigger": trigger,
                "reason": "Identical trigger and overlapping evidence_span",
                "page_id": page_id
            }
            event_logs.append(log_entry)
        else:
            # Add as new event
            new_ev = dict(ev)
            # Canonicalize participant names
            canon_parts = []
            for p in new_ev.get("participants", []):
                p_copy = dict(p)
                p_copy["name"] = alias_to_canonical.get(p_copy["name"], p_copy["name"])
                canon_parts.append(p_copy)
            new_ev["participants"] = canon_parts
            merged_events.append(new_ev)

    # Save event logs
    log_p = Path(log_file_path)
    log_p.parent.mkdir(parents=True, exist_ok=True)
    with open(log_p, "w", encoding="utf-8") as f:
        json.dump(event_logs, f, ensure_ascii=False, indent=2)

    return merged_events, event_logs


# ==============================================================================
# 4. Graph Validation (Cycles, Self-Loops & Chronological Constraints)
# ==============================================================================

def validate_knowledge_graph(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    events: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate and clean the knowledge graph:
      - Filters ungrounded items (is_supported == False)
      - Removes self-loops (head == tail)
      - Detects and breaks direct causal/temporal cycles (caused, resulted_in, preceded)
      - Validates chronological date ordering (causes cannot carry strictly later date than effects)

    Returns:
      - validated_graph: clean {entities, relations, events}
      - validation_report: statistics of removed invalid elements
    """
    report = {
        "self_loops_removed": 0,
        "causal_cycles_removed": 0,
        "chronological_violations_removed": 0,
        "unsupported_filtered": 0
    }

    # Filter ungrounded items if flag present
    clean_entities = [e for e in entities if e.get("is_supported", True)]
    report["unsupported_filtered"] += (len(entities) - len(clean_entities))

    # Map dates to events and entities
    entity_dates: Dict[str, Optional[int]] = {}
    for e in clean_entities:
        y = extract_year_from_text(e.get("name", ""))
        if y:
            entity_dates[e["name"]] = y

    for ev in events:
        ev_date = ev.get("date", "")
        y = extract_year_from_text(ev_date) or extract_year_from_text(ev.get("evidence_span", ""))
        if y:
            entity_dates[ev.get("trigger_name", "")] = y
            entity_dates[ev.get("event_type", "")] = y

    clean_relations = []
    # Directed graph for cycle detection on causal/temporal relations
    causal_types = {
        RelationType.caused.value,
        RelationType.resulted_in.value,
        RelationType.preceded.value
    }
    causal_adj: Dict[str, Set[str]] = defaultdict(set)

    def has_path(start: str, target: str, visited: Set[str]) -> bool:
        if start == target:
            return True
        visited.add(start)
        for neighbor in causal_adj.get(start, set()):
            if neighbor not in visited:
                if has_path(neighbor, target, visited):
                    return True
        return False

    for r in relations:
        if not r.get("is_supported", True):
            report["unsupported_filtered"] += 1
            continue

        head = r["head_text"]
        tail = r["tail_text"]
        rel_type = r["relation_type"]

        # Rule 1: Self-loop removal
        if head == tail:
            report["self_loops_removed"] += 1
            continue

        # Rule 2: Chronological validation
        # If head caused tail or preceded tail, head cannot occur in a later year than tail
        if rel_type in (RelationType.caused.value, RelationType.preceded.value):
            head_year = entity_dates.get(head) or extract_year_from_text(r.get("evidence_span", ""))
            tail_year = entity_dates.get(tail)
            if head_year and tail_year and head_year > tail_year:
                report["chronological_violations_removed"] += 1
                logger.warning(
                    f"Chronological violation rejected: '{head}' ({head_year}) cannot {rel_type} '{tail}' ({tail_year})"
                )
                continue

        # Rule 3: Causal cycle prevention
        if rel_type in causal_types:
            # If a directed path already exists from tail to head, adding head -> tail would create a cycle!
            if has_path(tail, head, set()):
                report["causal_cycles_removed"] += 1
                logger.warning(f"Causal cycle detected and broken: '{head}' -> '{tail}' creates a loop")
                continue
            causal_adj[head].add(tail)

        clean_relations.append(r)

    clean_events = [ev for ev in events if ev.get("is_supported", True)]

    validated_graph = {
        "entities": clean_entities,
        "relations": clean_relations,
        "events": clean_events
    }

    return validated_graph, report


# ==============================================================================
# 5. Dual Evaluation Helper (Raw Output vs Final Graph Comparison)
# ==============================================================================

def evaluate_raw_vs_final_graph(
    gold_items: List[Dict[str, Any]],
    raw_extraction_items: List[Dict[str, Any]],
    final_kg_items: List[Dict[str, Any]],
    gold_text: str = "",
    ocr_text: str = "",
    ocr_to_gold_char_map: Optional[Dict[int, Optional[int]]] = None
) -> Dict[str, Any]:
    """
    Evaluates and compares extraction performance before (raw_output)
    and after post-processing (final_kg) using Tier 1 and Tier 2 metrics.
    """
    # Raw output evaluation
    raw_t1 = evaluate_tier1_strict(gold_items, raw_extraction_items)
    raw_t2 = (
        evaluate_tier2_aligned(
            gold_items, raw_extraction_items, gold_text, ocr_text, ocr_to_gold_char_map
        )
        if ocr_to_gold_char_map
        else raw_t1
    )

    # Final post-processed KG evaluation
    final_t1 = evaluate_tier1_strict(gold_items, final_kg_items)
    final_t2 = (
        evaluate_tier2_aligned(
            gold_items, final_kg_items, gold_text, ocr_text, ocr_to_gold_char_map
        )
        if ocr_to_gold_char_map
        else final_t1
    )

    return {
        "raw": {
            "tier1": {"precision": raw_t1["precision"], "recall": raw_t1["recall"], "f1": raw_t1["f1"]},
            "tier2": {"precision": raw_t2["precision"], "recall": raw_t2["recall"], "f1": raw_t2["f1"]}
        },
        "final": {
            "tier1": {"precision": final_t1["precision"], "recall": final_t1["recall"], "f1": final_t1["f1"]},
            "tier2": {"precision": final_t2["precision"], "recall": final_t2["recall"], "f1": final_t2["f1"]}
        },
        "improvements": {
            "tier1_f1_gain": round(final_t1["f1"] - raw_t1["f1"], 4),
            "tier2_f1_gain": round(final_t2["f1"] - raw_t2["f1"], 4)
        }
    }
