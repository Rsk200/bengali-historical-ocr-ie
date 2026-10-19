#!/usr/bin/env python3
"""
Phase 5 & 5b: Real LLM Extraction & Knowledge Graph Construction Script
Executes:
  1. Real Gemini 3.8 Flash Extraction on S1 Chunks (p001 - p011) for gemini_raw and tesseract_raw.
  2. Grounding Verification (verify_evidence_spans) & unsupported_item_rate tracking.
  3. Post-Processing:
     - Inflection stripping & match_key standardisation
     - Logged entity resolution (logs/entity_merges.json)
     - Conservative event merging (logs/event_merges.json)
     - Graph validation (self-loops, causal cycles, chronological validation)
  4. Saves:
     - outputs/extractions/{source_name}/gemini/{page_id}.json
     - outputs/extractions/{source_name}/gemini/knowledge_graph_final.json
"""

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractor import GeminiLLMClient, LLMExtractionEngine, verify_evidence_spans
from src.postprocess import (
    deduplicate_relations,
    merge_events_conservatively,
    resolve_entities,
    validate_knowledge_graph
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
logger = logging.getLogger("dev_extraction_runner")


def run_extraction_and_kg(force: bool = False) -> Dict[str, Any]:
    load_dotenv(override=True)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured in .env!")

    client = GeminiLLMClient(api_key=api_key, model_name="gemini-3.5-flash")
    # Setting model_name="gemini" saves to outputs/extractions/{source_name}/gemini/{page_id}.json
    engine = LLMExtractionEngine(llm_client=client, model_name="gemini")

    sources = ["gemini_raw", "tesseract_raw"]
    pages = [f"p{i:03d}" for i in range(1, 12)]

    overall_results = {}

    for src in sources:
        logger.info(f"\n=======================================================")
        logger.info(f">>> STARTING EXTRACTION FOR SOURCE: {src}")
        logger.info(f"=======================================================")

        page_records = []
        raw_entities_all = []
        raw_relations_all = []
        raw_events_all = []

        total_items = 0
        total_unsupported = 0

        for pid in pages:
            chunk_file = Path(f"outputs/chunks/{src}/S1/{pid}_chunks.json")
            if not chunk_file.exists():
                logger.warning(f"Chunk file not found: {chunk_file}")
                continue

            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            out_json = Path(f"outputs/extractions/{src}/gemini/{pid}.json")

            if out_json.exists() and not force:
                logger.info(f"[{src}] Loading cached extraction for {pid}...")
                with open(out_json, "r", encoding="utf-8") as f:
                    rec = json.load(f)
            else:
                logger.info(f"[{src}] Extracting {pid} ({len(chunks)} chunks)...")
                rec = engine.extract_page_chunks(
                    chunks=chunks,
                    source_name=src,
                    page_id=pid,
                    split="dev",
                    force=force,
                    temperature=0.0
                )
                time.sleep(2)  # Polite rate limiting delay

            page_records.append(rec)
            raw_entities_all.extend(rec.get("entities", []))
            raw_relations_all.extend(rec.get("relations", []))
            raw_events_all.extend(rec.get("events", []))

            # Track unsupported rate
            for meta in rec.get("chunk_metadata", []):
                total_unsupported += meta.get("ungrounded_count", 0)
            p_items = len(rec.get("entities", [])) + len(rec.get("relations", [])) + len(rec.get("events", []))
            total_items += p_items

        unsupported_rate = total_unsupported / max(1, total_items)
        logger.info(f"[{src}] Raw Extraction Summary: {len(raw_entities_all)} entities, "
                    f"{len(raw_relations_all)} relations, {len(raw_events_all)} events. "
                    f"Unsupported item rate: {unsupported_rate:.4f} ({total_unsupported}/{total_items})")

        # -------------------------------------------------------------
        # Phase 5b Post-Processing & Knowledge Graph Construction
        # -------------------------------------------------------------
        logger.info(f"[{src}] Running Entity Resolution & Deduplication...")
        canon_entities, alias_map, entity_logs = resolve_entities(
            entities=raw_entities_all,
            page_id="dev_corpus",
            log_file_path="logs/entity_merges.json"
        )

        dedup_relations = deduplicate_relations(
            relations=raw_relations_all,
            alias_to_canonical=alias_map
        )

        merged_events, event_logs = merge_events_conservatively(
            events=raw_events_all,
            alias_to_canonical=alias_map,
            page_id="dev_corpus",
            log_file_path="logs/event_merges.json"
        )

        logger.info(f"[{src}] Running Knowledge Graph Validation (Self-loops, Cycles, Chronology)...")
        final_kg, val_report = validate_knowledge_graph(
            entities=canon_entities,
            relations=dedup_relations,
            events=merged_events
        )

        # Save Final Knowledge Graph
        kg_out_path = Path(f"outputs/extractions/{src}/gemini/knowledge_graph_final.json")
        with open(kg_out_path, "w", encoding="utf-8") as f:
            json.dump({
                "source_name": src,
                "split": "dev",
                "page_count": len(pages),
                "unsupported_item_rate": unsupported_rate,
                "postprocess_stats": {
                    "raw_entities": len(raw_entities_all),
                    "canonical_entities": len(final_kg["entities"]),
                    "entities_merged": len(raw_entities_all) - len(final_kg["entities"]),
                    "raw_relations": len(raw_relations_all),
                    "deduped_relations": len(final_kg["relations"]),
                    "relations_merged": len(raw_relations_all) - len(final_kg["relations"]),
                    "raw_events": len(raw_events_all),
                    "final_events": len(final_kg["events"]),
                    "events_merged": len(raw_events_all) - len(final_kg["events"]),
                    "validation_report": val_report
                },
                "graph": final_kg
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"[{src}] Final KG Saved to {kg_out_path}")
        logger.info(f"[{src}] Final Graph: {len(final_kg['entities'])} entities, "
                    f"{len(final_kg['relations'])} relations, {len(final_kg['events'])} events")

        overall_results[src] = {
            "raw": {
                "entities": raw_entities_all,
                "relations": raw_relations_all,
                "events": raw_events_all,
                "unsupported_rate": unsupported_rate
            },
            "final": final_kg,
            "validation_report": val_report,
            "entity_merges_count": len(entity_logs),
            "event_merges_count": len(event_logs)
        }

    return overall_results


if __name__ == "__main__":
    run_extraction_and_kg(force=False)
