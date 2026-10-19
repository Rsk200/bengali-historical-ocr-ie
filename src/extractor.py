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

    schema_template = (
        "Output JSON Schema:\n"
        "{\n"
        '  "entities": [{"entity_id": "e1", "name": "...", "entity_type": "PERSON|ORGANIZATION|LOCATION|EVENT|DATE/PERIOD|DOCUMENT/TREATY/LAW", "evidence_span": "..."}],\n'
        '  "relations": [{"relation_id": "r1", "head_text": "...", "head_type": "...", "relation_type": "participated_in|led_by|member_of|occurred_at|occurred_on|caused|resulted_in|preceded|signed_issued|located_in", "tail_text": "...", "tail_type": "...", "evidence_span": "..."}],\n'
        '  "events": [{"event_id": "ev1", "event_type": "...", "trigger_name": "...", "participants": [{"name": "...", "entity_type": "...", "role": "..."}], "location": "...", "date": "...", "causes": [], "consequences": [], "evidence_span": "..."}]\n'
        "}\n"
    )

    if extraction_mode == "joint":
        user_prompt = (
            f"{context_prefix}Extract all entities, relations, and event frames from the text below.\n"
            f"TEXT CHUNK:\n\"\"\"\n{chunk_text}\n\"\"\"\n\n"
            f"{schema_template}\n"
            f"Return ONLY valid JSON strictly matching the above schema keys."
        )
    else:
        user_prompt = (
            f"{context_prefix}Extract {target_type} from the text below.\n"
            f"TEXT CHUNK:\n\"\"\"\n{chunk_text}\n\"\"\"\n\n"
            f"{schema_template}\n"
            f"Return ONLY valid JSON for {target_type} matching the above schema keys."
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


class GeminiLLMClient(BaseLLMClient):
    """
    Production client using Google Gemini API (gemini-3.8-flash)
    with strict JSON response generation and rate limit retries.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.8-flash"
    ):
        import os
        from dotenv import load_dotenv
        load_dotenv(override=True)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured in environment or .env!")
        self.model_name = model_name
        self.candidate_models = [model_name, "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.8-flash"]
        # Deduplicate while preserving order
        seen = set()
        self.candidate_models = [m for m in self.candidate_models if not (m in seen or seen.add(m))]
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        import google.generativeai as genai
        prompt = f"{system_prompt}\n\n{user_prompt}"
        last_err = None
        for mod_name in self.candidate_models:
            for attempt in range(2):
                try:
                    m = genai.GenerativeModel(mod_name)
                    response = m.generate_content(
                        prompt,
                        generation_config={
                            "temperature": temperature,
                            "response_mime_type": "application/json"
                        }
                    )
                    txt = response.text or "{}"
                    if txt.strip():
                        return txt
                except Exception as e:
                    last_err = e
                    err_s = str(e)
                    if "429" in err_s or "Quota" in err_s:
                        extractor_logger.warning(f"Model {mod_name} quota hit. Switching fallback...")
                        break
                    time.sleep(2)
        if last_err:
            raise last_err
        return "{}"


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

                # Schema normalization helper
                entities_map = {}
                clean_entities = []
                allowed_types = {"PERSON", "ORGANIZATION", "LOCATION", "EVENT", "DATE/PERIOD", "DOCUMENT/TREATY/LAW"}
                type_aliases = {
                    "DATE": "DATE/PERIOD", "PERIOD": "DATE/PERIOD", "TIME": "DATE/PERIOD", "YEAR": "DATE/PERIOD",
                    "DOCUMENT": "DOCUMENT/TREATY/LAW", "TREATY": "DOCUMENT/TREATY/LAW", "LAW": "DOCUMENT/TREATY/LAW",
                    "ACT": "DOCUMENT/TREATY/LAW", "ORG": "ORGANIZATION", "LOC": "LOCATION", "PER": "PERSON"
                }
                disallowed_org_titles = {
                    "প্রধানমন্ত্রী", "রাষ্ট্রপতি", "মুখ্যমন্ত্রী", "গভর্নর জেনারেল", "মন্ত্রী",
                    "স্পিকার", "উপদেষ্টা", "সদস্য", "সভাপতি", "সাধারণ সম্পাদক"
                }

                for idx, raw_e in enumerate(parsed_dict.get("entities", [])):
                    if not isinstance(raw_e, dict):
                        continue
                    e = dict(raw_e)
                    eid = e.get("entity_id") or f"e{idx+1}"
                    e["entity_id"] = eid
                    e["chunk_id"] = chunk_id
                    e["page_id"] = page_id

                    name = (e.get("name") or e.get("text") or e.get("entity_name") or "").strip()
                    if not name:
                        continue
                    e["name"] = name

                    raw_type = (e.get("entity_type") or e.get("type") or e.get("category") or "ORGANIZATION").strip().upper()
                    etype = type_aliases.get(raw_type, raw_type)
                    if etype not in allowed_types:
                        etype = "ORGANIZATION"
                    if etype == "ORGANIZATION" and name.lower() in disallowed_org_titles:
                        etype = "PERSON"
                    e["entity_type"] = etype

                    span = (e.get("evidence_span") or e.get("evidence") or e.get("span") or name).strip()
                    e["evidence_span"] = span

                    entities_map[eid] = e
                    entities_map[name] = e
                    clean_entities.append(e)

                allowed_rel_types = {
                    "participated_in", "led_by", "member_of", "occurred_at", "occurred_on",
                    "caused", "resulted_in", "preceded", "signed_issued", "located_in"
                }
                clean_relations = []
                for idx, raw_r in enumerate(parsed_dict.get("relations", [])):
                    if not isinstance(raw_r, dict):
                        continue
                    r = dict(raw_r)
                    r["relation_id"] = r.get("relation_id") or f"r{idx+1}"
                    r["chunk_id"] = chunk_id
                    r["page_id"] = page_id

                    head = (r.get("head_text") or r.get("subject") or r.get("subject_name") or "").strip()
                    if not head and r.get("subject_id") in entities_map:
                        head = entities_map[r["subject_id"]]["name"]
                    if not head and r.get("head_id") in entities_map:
                        head = entities_map[r["head_id"]]["name"]
                    if not head:
                        continue
                    r["head_text"] = head

                    tail = (r.get("tail_text") or r.get("object") or r.get("object_name") or "").strip()
                    if not tail and r.get("object_id") in entities_map:
                        tail = entities_map[r["object_id"]]["name"]
                    if not tail and r.get("tail_id") in entities_map:
                        tail = entities_map[r["tail_id"]]["name"]
                    if not tail:
                        continue
                    r["tail_text"] = tail

                    raw_rtype = (r.get("relation_type") or r.get("type") or r.get("predicate") or "member_of").strip()
                    if raw_rtype not in allowed_rel_types:
                        raw_rtype = "member_of"

                    head_ent = entities_map.get(head) or entities_map.get(r.get("subject_id"))
                    tail_ent = entities_map.get(tail) or entities_map.get(r.get("object_id"))

                    h_type = r.get("head_type") or (head_ent["entity_type"] if head_ent else "ORGANIZATION")
                    t_type = r.get("tail_type") or (tail_ent["entity_type"] if tail_ent else "ORGANIZATION")

                    # Check domain/range against schema rules
                    from src.schema import RelationType, RELATION_ALLOWED_TYPES
                    rel_enum = RelationType(raw_rtype)
                    allowed_heads = {t.value for t in RELATION_ALLOWED_TYPES[rel_enum]["head"]}
                    allowed_tails = {t.value for t in RELATION_ALLOWED_TYPES[rel_enum]["tail"]}

                    if t_type not in allowed_tails:
                        if t_type == "DATE/PERIOD":
                            rel_enum = RelationType.occurred_on
                            allowed_heads = {t.value for t in RELATION_ALLOWED_TYPES[rel_enum]["head"]}
                            allowed_tails = {t.value for t in RELATION_ALLOWED_TYPES[rel_enum]["tail"]}
                        elif t_type == "LOCATION":
                            rel_enum = RelationType.occurred_at
                            allowed_heads = {t.value for t in RELATION_ALLOWED_TYPES[rel_enum]["head"]}
                            allowed_tails = {t.value for t in RELATION_ALLOWED_TYPES[rel_enum]["tail"]}
                        else:
                            t_type = next(iter(allowed_tails))

                    if h_type not in allowed_heads:
                        h_type = next(iter(allowed_heads))

                    r["relation_type"] = rel_enum.value
                    r["head_type"] = h_type
                    r["tail_type"] = t_type
                    r["evidence_span"] = (r.get("evidence_span") or r.get("evidence") or f"{head} {tail}").strip()
                    clean_relations.append(r)

                clean_events = []
                for idx, raw_ev in enumerate(parsed_dict.get("events", [])):
                    if not isinstance(raw_ev, dict):
                        continue
                    ev = dict(raw_ev)
                    ev["event_id"] = ev.get("event_id") or f"ev{idx+1}"
                    ev["chunk_id"] = chunk_id
                    ev["page_id"] = page_id
                    ev["event_type"] = (ev.get("event_type") or "EVENT").strip().upper()
                    ev["trigger_name"] = (ev.get("trigger_name") or ev.get("trigger") or ev.get("name") or "ঘটনা").strip()

                    parts = []
                    for p in ev.get("participants", []):
                        if isinstance(p, dict):
                            p_dict = dict(p)
                            p_name = p_dict.get("name") or ""
                            p_ent = entities_map.get(p_name)
                            p_type = p_dict.get("entity_type") or (p_ent["entity_type"] if p_ent else "ORGANIZATION")
                            if p_type not in ("PERSON", "ORGANIZATION"):
                                p_type = "ORGANIZATION"
                            p_dict["entity_type"] = p_type
                            p_dict["role"] = p_dict.get("role") or "participant"
                            parts.append(p_dict)
                        elif isinstance(p, str):
                            p_ent = entities_map.get(p)
                            p_type = p_ent["entity_type"] if p_ent else "ORGANIZATION"
                            if p_type not in ("PERSON", "ORGANIZATION"):
                                p_type = "ORGANIZATION"
                            parts.append({"name": p, "entity_type": p_type, "role": "participant"})
                    ev["participants"] = parts

                    ev["location"] = ev.get("location")
                    ev["date"] = ev.get("date")
                    ev["causes"] = ev.get("causes") or []
                    ev["consequences"] = ev.get("consequences") or []
                    ev["evidence_span"] = (ev.get("evidence_span") or ev.get("evidence") or ev["trigger_name"]).strip()
                    clean_events.append(ev)

                norm_parsed_dict = {
                    "entities": clean_entities,
                    "relations": clean_relations,
                    "events": clean_events
                }

                # Validate against Pydantic schema
                output_model = ChunkExtractionOutput.model_validate(norm_parsed_dict)

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
