#!/usr/bin/env python3
"""
LLM Information Extraction Engine
Supports:
  - Strict Pydantic schema validation against ChunkExtractionOutput
  - Prompt ablation modes (zero_shot vs few_shot; joint vs per_type)
  - Automatic retry with error feedback (up to max_retries=3)
  - Execution logging to logs/llm_extraction.log
  - Verbatim evidence_span grounding verification & hallucination tracking
  - Resumable per-page JSON caching with test-set protection guard
  - Offline MockLLMClient for testing and CI/CD
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import yaml

from src.normalise import normalise_bangla_text
from src.schema import (
    ChunkExtractionOutput,
    EntityType,
    ExtractedEntity,
    ExtractedEventFrame,
    ExtractedRelation,
    RelationType
)

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
extractor_logger = logging.getLogger("llm_extractor")
extractor_logger.setLevel(logging.INFO)
if not extractor_logger.handlers:
    fh = logging.FileHandler(log_dir / "llm_extraction.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    extractor_logger.addHandler(fh)


# ==============================================================================
# 1. Prompt Builder
# ==============================================================================

SYSTEM_PROMPT_BASE = """You are an expert Historical Information Extraction system analyzing Bengali National Curriculum (NCTB Class 9-10) textbooks.
Your objective is to extract historical entities, directed relations, and event frames with 100% precision.

CRITICAL INSTRUCTIONS:
1. EXTRACT ONLY FACTS EXPLICITLY STATED in the provided text chunk. DO NOT use external pre-training knowledge or memorised historical details.
2. For EVERY extracted entity, relation, and event, you MUST provide an 'evidence_span': an EXACT, VERBATIM substring copied directly from the text chunk proving the fact.
3. Output MUST strictly conform to the provided JSON schema.
4. BARE ROLES/TITLES alone (e.g. 'প্রধানমন্ত্রী', 'রাষ্ট্রপতি', 'মুখ্যমন্ত্রী') CANNOT be extracted as ORGANIZATION nodes.

Allowed Entity Types:
- PERSON, ORGANIZATION, LOCATION, EVENT, DATE/PERIOD, DOCUMENT/TREATY/LAW

Allowed Relation Types (Subject -> Object):
- participated_in (PERSON/ORGANIZATION -> EVENT)
- led_by (ORGANIZATION/EVENT -> PERSON)
- member_of (PERSON -> ORGANIZATION)
- occurred_at (EVENT -> LOCATION)
- occurred_on (EVENT/DOCUMENT/TREATY/LAW -> DATE/PERIOD)
- caused (EVENT/DOCUMENT/TREATY/LAW -> EVENT)
- resulted_in (EVENT -> EVENT/DOCUMENT/TREATY/LAW/ORGANIZATION)
- preceded (EVENT -> EVENT)
- signed_issued (PERSON/ORGANIZATION -> DOCUMENT/TREATY/LAW)
- located_in (LOCATION/ORGANIZATION -> LOCATION)
"""

FEW_SHOT_EXAMPLE = """
Example:
Input Text:
"১৯৪৮ সালের ২৩শে ফেব্রুয়ারি করাচিতে পাকিস্তান গণপরিষদের প্রথম অধিবেশন শুরু হয়। গণপরিষদের সদস্য ধীরেন্দ্রনাথ দত্ত পরিষদের ভাষা হিসেবে উর্দু ও ইংরেজির পাশাপাশি বাংলা ব্যবহারের দাবি জানান।"

Output JSON:
{
  "entities": [
    {
      "entity_id": "e1",
      "name": "১৯৪৮ সালের ২৩শে ফেব্রুয়ারি",
      "entity_type": "DATE/PERIOD",
      "evidence_span": "১৯৪৮ সালের ২৩শে ফেব্রুয়ারি করাচিতে"
    },
    {
      "entity_id": "e2",
      "name": "করাচি",
      "entity_type": "LOCATION",
      "evidence_span": "করাচিতে পাকিস্তান গণপরিষদের"
    },
    {
      "entity_id": "e3",
      "name": "পাকিস্তান গণপরিষদ",
      "entity_type": "ORGANIZATION",
      "evidence_span": "পাকিস্তান গণপরিষদের প্রথম অধিবেশন"
    },
    {
      "entity_id": "e4",
      "name": "ধীরেন্দ্রনাথ দত্ত",
      "entity_type": "PERSON",
      "evidence_span": "গণপরিষদের সদস্য ধীরেন্দ্রনাথ দত্ত"
    }
  ],
  "relations": [
    {
      "relation_id": "r1",
      "head_text": "ধীরেন্দ্রনাথ দত্ত",
      "head_type": "PERSON",
      "relation_type": "member_of",
      "tail_text": "পাকিস্তান গণপরিষদ",
      "tail_type": "ORGANIZATION",
      "evidence_span": "গণপরিষদের সদস্য ধীরেন্দ্রনাথ দত্ত"
    }
  ],
  "events": [
    {
      "event_id": "ev1",
      "event_type": "MEETING",
      "trigger_name": "অধিবেশন",
      "participants": [
        {"name": "পাকিস্তান গণপরিষদ", "entity_type": "ORGANIZATION", "role": "organizer"},
        {"name": "ধীরেন্দ্রনাথ দত্ত", "entity_type": "PERSON", "role": "participant"}
      ],
      "location": "করাচি",
      "date": "১৯৪৮ সালের ২৩শে ফেব্রুয়ারি",
      "causes": [],
      "consequences": ["বাংলা ব্যবহারের দাবি উত্থাপন"],
      "evidence_span": "করাচিতে পাকিস্তান গণপরিষদের প্রথম অধিবেশন শুরু হয়"
    }
  ]
}
"""


def build_extraction_prompt(
    chunk_text: str,
    heading_path: Optional[List[str]] = None,
    shot_mode: str = "zero_shot",
    extraction_mode: str = "joint",
    target_type: Optional[str] = None
) -> Tuple[str, str]:
    """
    Constructs system and user prompts supporting:
      - shot_mode: 'zero_shot' or 'few_shot'
      - extraction_mode: 'joint' (all together) or 'per_type' ('entities', 'relations', 'events')
    """
    system_prompt = SYSTEM_PROMPT_BASE

    if shot_mode == "few_shot":
        system_prompt += "\n" + FEW_SHOT_EXAMPLE

    context_prefix = ""
    if heading_path:
        context_prefix = f"Section Context: {' > '.join(heading_path)}\n\n"

    if extraction_mode == "joint":
        user_prompt = (
            f"{context_prefix}Extract all entities, relations, and event frames from the text below.\n"
            f"TEXT CHUNK:\n\"\"\"\n{chunk_text}\n\"\"\"\n\n"
            f"Return ONLY valid JSON matching the ChunkExtractionOutput schema."
        )
    else:
        user_prompt = (
            f"{context_prefix}Extract {target_type} from the text below.\n"
            f"TEXT CHUNK:\n\"\"\"\n{chunk_text}\n\"\"\"\n\n"
            f"Return ONLY valid JSON for {target_type} matching the schema."
        )

    return system_prompt, user_prompt


# ==============================================================================
# 2. Verbatim Evidence Span Verification
# ==============================================================================

def verify_evidence_spans(
    output: ChunkExtractionOutput,
    chunk_text: str
) -> Tuple[ChunkExtractionOutput, List[Dict[str, Any]], float]:
    """
    Verify that every evidence_span in extracted entities, relations, and events
    is grounded verbatim in the source chunk text.
    Flags ungrounded items as unsupported (hallucinations) and computes unsupported_item_rate.
    """
    norm_chunk = normalise_bangla_text(chunk_text, digits_target="bangla")
    unsupported_items = []
    total_items = len(output.entities) + len(output.relations) + len(output.events)

    def is_grounded(span_text: str) -> bool:
        if not span_text or not span_text.strip():
            return False
        if span_text in chunk_text:
            return True
        norm_span = normalise_bangla_text(span_text, digits_target="bangla")
        return norm_span in norm_chunk

    # Check entities
    for ent in output.entities:
        if not is_grounded(ent.evidence_span):
            unsupported_items.append({"category": "ENTITY", "id": ent.entity_id, "span": ent.evidence_span})

    # Check relations
    for rel in output.relations:
        if not is_grounded(rel.evidence_span):
            unsupported_items.append({"category": "RELATION", "id": rel.relation_id, "span": rel.evidence_span})

    # Check events
    for ev in output.events:
        if not is_grounded(ev.evidence_span):
            unsupported_items.append({"category": "EVENT", "id": ev.event_id, "span": ev.evidence_span})

    rate = len(unsupported_items) / max(1, total_items)
    return output, unsupported_items, round(rate, 4)


# ==============================================================================
# 3. LLM Client Wrapper with Retry & Resumable Caching
# ==============================================================================

class BaseLLMClient:
    """Base client interface for LLM extraction calls."""
    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """
    Deterministic mock client for offline testing and pipeline verification.
    Supports injecting deliberate validation errors for testing the retry loop.
    """
    def __init__(self, response_generator: Optional[Callable[[str, str], str]] = None):
        self.response_generator = response_generator
        self.call_count = 0

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        self.call_count += 1
        if self.response_generator:
            return self.response_generator(system_prompt, user_prompt)
        # Default minimal valid JSON
        return json.dumps({
            "entities": [
                {
                    "entity_id": "e1",
                    "name": "ধীরেন্দ্রনাথ দত্ত",
                    "entity_type": "PERSON",
                    "evidence_span": "ধীরেন্দ্রনাথ দত্ত পরিষদের ভাষা হিসেবে"
                }
            ],
            "relations": [],
            "events": []
        })


class LLMExtractionEngine:
    """
    Manages schema validation, retry with error feedback, logging,
    resumable per-page caching, and test set protection.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        model_name: str = "gemini-1.5-pro",
        config_path: Union[str, Path] = "configs/config.yaml",
        output_base_dir: Union[str, Path] = "outputs/extractions"
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.config_path = Path(config_path)
        self.output_base_dir = Path(output_base_dir)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def extract_chunk(
        self,
        chunk_dict: Dict[str, Any],
        temperature: float = 0.0,
        max_retries: int = 3,
        shot_mode: str = "zero_shot",
        extraction_mode: str = "joint"
    ) -> Tuple[ChunkExtractionOutput, Dict[str, Any]]:
        """
        Extract facts from a single chunk with automatic retry upon schema validation failure.
        """
        chunk_text = chunk_dict["text"]
        heading_path = chunk_dict.get("heading_path", [])
        chunk_id = chunk_dict.get("chunk_id", "c001")
        page_id = chunk_dict.get("page_id", "p001")

        sys_prompt, user_prompt = build_extraction_prompt(
            chunk_text=chunk_text,
            heading_path=heading_path,
            shot_mode=shot_mode,
            extraction_mode=extraction_mode
        )

        attempts = 0
        last_error = ""
        current_user_prompt = user_prompt

        while attempts < max_retries:
            attempts += 1
            t_start = time.perf_counter()
            try:
                raw_response = self.llm_client.generate_json(
                    sys_prompt, current_user_prompt, temperature=temperature
                )
                latency = round(time.perf_counter() - t_start, 3)

                # Clean markdown JSON block formatting if present
                clean_json_str = raw_response.strip()
                if clean_json_str.startswith("```json"):
                    clean_json_str = clean_json_str[7:]
                if clean_json_str.startswith("```"):
                    clean_json_str = clean_json_str[3:]
                if clean_json_str.endswith("```"):
                    clean_json_str = clean_json_str[:-3]
                clean_json_str = clean_json_str.strip()

                parsed_dict = json.loads(clean_json_str)

                # Populate chunk_id and page_id if missing
                for item_list in ("entities", "relations", "events"):
                    for item in parsed_dict.get(item_list, []):
                        if not item.get("chunk_id"):
                            item["chunk_id"] = chunk_id
                        if not item.get("page_id"):
                            item["page_id"] = page_id

                # Validate against Pydantic schema
                output_model = ChunkExtractionOutput.model_validate(parsed_dict)

                # Verify verbatim evidence spans
                verified_output, ungrounded, ungrounded_rate = verify_evidence_spans(
                    output_model, chunk_text
                )

                meta = {
                    "attempts": attempts,
                    "latency_sec": latency,
                    "status": "SUCCESS",
                    "ungrounded_count": len(ungrounded),
                    "unsupported_item_rate": ungrounded_rate
                }
                extractor_logger.info(
                    f"[{self.model_name}] Extracted {chunk_id}: {len(output_model.entities)} entities, "
                    f"{len(output_model.relations)} relations, {len(output_model.events)} events in {latency}s (attempt {attempts})"
                )
                return verified_output, meta

            except Exception as e:
                latency = round(time.perf_counter() - t_start, 3)
                last_error = str(e)
                extractor_logger.warning(
                    f"[{self.model_name}] Attempt {attempts}/{max_retries} failed for {chunk_id} in {latency}s: {last_error}"
                )
                # Feed the validation error back to the LLM for the next retry attempt
                current_user_prompt = (
                    f"{user_prompt}\n\n"
                    f"PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{last_error}\n"
                    f"Please correct the error and return ONLY valid JSON matching the schema."
                )

        raise ValueError(f"Extraction failed after {max_retries} attempts for {chunk_id}. Last error: {last_error}")

    def extract_page_chunks(
        self,
        chunks: List[Dict[str, Any]],
        source_name: str,
        page_id: str,
        split: str = "dev",
        force: bool = False,
        allow_test: bool = False,
        temperature: float = 0.0,
        shot_mode: str = "zero_shot",
        extraction_mode: str = "joint"
    ) -> Dict[str, Any]:
        """
        Extract all chunks for a page, enforcing held-out test set protection
        and saving results to outputs/extractions/{source_name}/{model_name}/{page_id}.json.
        """
        # Strict protection guard for held-out test split
        if split == "test" and not allow_test:
            if self.config.get("test_set_read_only", True):
                raise PermissionError(
                    f"Held-out test set protection is ACTIVE! "
                    f"Running LLM extraction on 'test' split is prohibited during development. "
                    f"Pass allow_test=True explicitly for final benchmark."
                )

        out_dir = self.output_base_dir / source_name / self.model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{page_id}.json"

        if out_path.exists() and not force:
            extractor_logger.info(f"[{self.model_name}] Skipping {page_id}: cached at {out_path}")
            with open(out_path, "r", encoding="utf-8") as f:
                return json.load(f)

        all_entities = []
        all_relations = []
        all_events = []
        chunk_metas = []

        for chunk in chunks:
            out_model, meta = self.extract_chunk(
                chunk_dict=chunk,
                temperature=temperature,
                shot_mode=shot_mode,
                extraction_mode=extraction_mode
            )
            all_entities.extend([e.model_dump() for e in out_model.entities])
            all_relations.extend([r.model_dump() for r in out_model.relations])
            all_events.extend([ev.model_dump() for ev in out_model.events])
            chunk_metas.append(meta)

        page_record = {
            "page_id": page_id,
            "source_name": source_name,
            "model_name": self.model_name,
            "split": split,
            "chunk_count": len(chunks),
            "entities": all_entities,
            "relations": all_relations,
            "events": all_events,
            "chunk_metadata": chunk_metas
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(page_record, f, ensure_ascii=False, indent=2)

        return page_record
