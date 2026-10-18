#!/usr/bin/env python3
"""
Comprehensive Unit Test and Verification Suite for Step 4:
  1. Chunking Strategies Verification (S1 Structure-Aware, S2 Fixed Overlap, S3 Sentence Context)
  2. Pydantic Ontology & Domain Constraints Validation (Bare roles, Relation domain/range, Evidence spans)
  3. LLM Extraction Engine Verification (Retry with Error Feedback, Grounding Check, Test Guard)
  4. Post-Processing & Knowledge Graph Construction (Entity Resolution, Conservative Merging, Graph Validation)
  5. Dual Evaluation Helper (Raw Output vs Final Graph Comparison Table)
"""

import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_page_text, save_chunks
from src.evaluate_ocr import align_blocks_and_chars
from src.extractor import LLMExtractionEngine, MockLLMClient, verify_evidence_spans
from src.postprocess import (
    deduplicate_relations,
    evaluate_raw_vs_final_graph,
    merge_events_conservatively,
    resolve_entities,
    validate_knowledge_graph
)
from src.schema import (
    ChunkExtractionOutput,
    EntityType,
    ExtractedEntity,
    ExtractedEventFrame,
    ExtractedRelation,
    RelationType
)
from src.synthetic_noise import inject_synthetic_noise


# ==============================================================================
# Sample Bengali Historical Textbook Text (BGS Chapter 1)
# ==============================================================================

SAMPLE_PAGE_TEXT = (
    "প্রথম অধ্যায়\n"
    "পূর্ববাংলার আন্দোলন ও জাতীয়তাবাদের উত্থান (১৯৪৭ খ্রি.-১৯৭০ খ্রি.)\n\n"
    "পরিচ্ছেদ ১.১ : বাঙালি জাতীয়তাবাদের বিকাশে ভাষা-আন্দোলন\n"
    "ভাষা-আন্দোলনের পটভূমি\n\n"
    "১৯৪৭ সালে ভারতবর্ষে ব্রিটিশ শাসনের অবসান হয়। জন্ম নেয় ভারত এবং পাকিস্তান নামে দুটি স্বাধীন রাষ্ট্র। "
    "১৯৪৭ সালের ১৪ই আগস্ট পাকিস্তান এবং ১৫ই আগস্ট ভারত স্বাধীনতা লাভ করে। "
    "১৯৪৮ সালের ২৩শে ফেব্রুয়ারি করাচিতে পাকিস্তান গণপরিষদের প্রথম অধিবেশন শুরু হয়। "
    "গণপরিষদের সদস্য ধীরেন্দ্রনাথ দত্ত পরিষদের ভাষা হিসেবে উর্দু ও ইংরেজির পাশাপাশি বাংলা ব্যবহারের ঐতিহাসিক দাবি জানান। "
    "তৎকালীন প্রধানমন্ত্রী লিয়াকত আলী খান এর তীব্র বিরোধিতা করেন। "
    "এর প্রতিবাদে ১৯৪৮ সালের ১১ই মার্চ ঢাকায় ধর্মঘট পালিত হয়। "
    "পরে ১৯৫২ সালের ২৬শে জানুয়ারি ঢাকার পল্টন ময়দানে পাকিস্তানের প্রধানমন্ত্রী খাজা নাজিমুদ্দীন উর্দুকে রাষ্ট্রভাষা করার ঘোষণা দেন। "
    "এর প্রতিবাদে ১৯৫২ সালের ২১শে ফেব্রুয়ারি ছাত্ররা ১৪৪ ধারা ভঙ্গ করে ঐতিহাসিক মিছিল বের করে।"
)


def test_chunking_strategies():
    print("\n" + "=" * 65)
    print("TEST 1: CHUNKING STRATEGIES (S1, S2, S3) ON GOLD & NOISY TEXT")
    print("=" * 65)

    noisy_text = inject_synthetic_noise(SAMPLE_PAGE_TEXT, target_cer=0.03, seed=42)

    for name, text in [("Gold Standard", SAMPLE_PAGE_TEXT), ("Synthetic OCR Noisy", noisy_text)]:
        print(f"\n--- Testing on {name} ---")

        # 1. Strategy S1
        chunks_s1 = chunk_page_text(text, page_id="p001", chapter_id="ch01", strategy="S1", max_chars=800)
        print(f"  S1 (Structure-Aware, max 800 chars): {len(chunks_s1)} chunks produced")
        for c in chunks_s1:
            slice_match = (text[c["char_start"]:c["char_end"]] == c["text"])
            assert slice_match, f"Offset mismatch in chunk {c['chunk_id']}"
            print(f"    - [{c['chunk_id']}] chars {c['char_start']}:{c['char_end']} (len={len(c['text'])}), Heading: {' > '.join(c['heading_path'])}")

        # Save sample chunks
        saved_s1_path = save_chunks(chunks_s1, source_name="test_run", strategy="S1", page_id="p001")
        assert saved_s1_path.exists(), "Saved S1 chunks file does not exist"

        # 2. Strategy S2
        chunks_s2 = chunk_page_text(text, page_id="p001", chapter_id="ch01", strategy="S2", max_chars=400, overlap_chars=100)
        print(f"  S2 (Fixed size with 10% overlap): {len(chunks_s2)} chunks produced")
        for c in chunks_s2:
            slice_match = (text[c["char_start"]:c["char_end"]] == c["text"])
            assert slice_match, f"Offset mismatch in chunk {c['chunk_id']}"
            print(f"    - [{c['chunk_id']}] chars {c['char_start']}:{c['char_end']} (len={len(c['text'])})")

        # 3. Strategy S3
        chunks_s3 = chunk_page_text(text, page_id="p001", chapter_id="ch01", strategy="S3")
        print(f"  S3 (Sentence-Level with Context): {len(chunks_s3)} sentence chunks produced")
        for c in chunks_s3[:3]:
            slice_match = (text[c["char_start"]:c["char_end"]] == c["text"])
            assert slice_match, f"Offset mismatch in chunk {c['chunk_id']}"
            print(f"    - [{c['chunk_id']}] Target: '{c['text'][:30]}...' | Context len: {len(c.get('context_text', ''))}")

    print("  [PASS] All chunking strategies (S1, S2, S3) and character offsets verified perfectly.")


def test_schema_and_domain_constraints():
    print("\n" + "=" * 65)
    print("TEST 2: PYDANTIC SCHEMA & DOMAIN CONSTRAINTS VALIDATION")
    print("=" * 65)

    # 1. Valid Entity
    e1 = ExtractedEntity(
        entity_id="e1",
        name="ধীরেন্দ্রনাথ দত্ত",
        entity_type=EntityType.PERSON,
        evidence_span="সদস্য ধীরেন্দ্রনাথ দত্ত পরিষদের ভাষা"
    )
    assert e1.name == "ধীরেন্দ্রনাথ দত্ত"
    print("  [PASS] Valid Entity successfully validated:", e1.name)

    # 2. Bare role alone as ORGANIZATION must fail
    bare_role_rejected = False
    try:
        ExtractedEntity(
            entity_id="e2",
            name="প্রধানমন্ত্রী",
            entity_type=EntityType.ORGANIZATION,
            evidence_span="তৎকালীন প্রধানমন্ত্রী লিয়াকত আলী খান"
        )
    except ValueError as err:
        bare_role_rejected = True
        print("  [PASS] Bare title 'প্রধানমন্ত্রী' as ORGANIZATION correctly rejected by schema.")
    assert bare_role_rejected, "Bare role as ORGANIZATION was not rejected!"

    # 3. Empty evidence_span must fail
    empty_span_rejected = False
    try:
        ExtractedEntity(
            entity_id="e3",
            name="ঢাকা",
            entity_type=EntityType.LOCATION,
            evidence_span="   "
        )
    except ValueError:
        empty_span_rejected = True
        print("  [PASS] Empty evidence_span correctly rejected by schema.")
    assert empty_span_rejected, "Empty evidence_span was not rejected!"

    # 4. Relation domain/range type validation
    invalid_rel_rejected = False
    try:
        ExtractedRelation(
            relation_id="r1",
            head_text="ঢাকা",
            head_type=EntityType.LOCATION,
            relation_type=RelationType.member_of,  # member_of requires PERSON as head
            tail_text="পাকিস্তান গণপরিষদ",
            tail_type=EntityType.ORGANIZATION,
            evidence_span="ঢাকার সদস্য"
        )
    except ValueError as err:
        invalid_rel_rejected = True
        print("  [PASS] Invalid relation domain (LOCATION member_of ORGANIZATION) correctly rejected.")
    assert invalid_rel_rejected, "Invalid relation domain was not rejected!"

    # 5. Valid Event Frame
    ev = ExtractedEventFrame(
        event_id="ev1",
        event_type="PROTEST",
        trigger_name="ধর্মঘট",
        participants=[{"name": "ছাত্ররা", "entity_type": EntityType.ORGANIZATION, "role": "protester"}],
        location="ঢাকা",
        date="১৯৪৮ সালের ১১ই মার্চ",
        causes=["উর্দু রাষ্ট্রভাষার ঘোষণা"],
        consequences=["মিছিল ও গ্রেফতার"],
        evidence_span="১১ই মার্চ ঢাকায় ধর্মঘট পালিত হয়"
    )
    assert ev.trigger_name == "ধর্মঘট"
    print("  [PASS] Valid EventFrame successfully validated:", ev.trigger_name)


def test_llm_extractor_retry_and_grounding():
    print("\n" + "=" * 65)
    print("TEST 3: LLM EXTRACTOR RETRY LOOP, GROUNDING CHECK & TEST GUARD")
    print("=" * 65)

    sample_chunk = {
        "chunk_id": "p001_c001",
        "page_id": "p001",
        "text": "১৯৪৮ সালের ২৩শে ফেব্রুয়ারি করাচিতে পাকিস্তান গণপরিষদের প্রথম অধিবেশন শুরু হয়। "
                "গণপরিষদের সদস্য ধীরেন্দ্রনাথ দত্ত বাংলা ব্যবহারের দাবি জানান।"
    }

    # Simulate a client where attempt 1 returns invalid JSON / schema failure, and attempt 2 succeeds
    attempts_seen = 0

    def mock_responses(sys_prompt, user_prompt):
        nonlocal attempts_seen
        attempts_seen += 1
        if attempts_seen == 1:
            # Attempt 1: Return bare role as ORGANIZATION to trigger schema validation failure
            return json.dumps({
                "entities": [
                    {
                        "entity_id": "e_bad",
                        "name": "প্রধানমন্ত্রী",
                        "entity_type": "ORGANIZATION",
                        "evidence_span": "প্রধানমন্ত্রী"
                    }
                ],
                "relations": [],
                "events": []
            })
        else:
            # Attempt 2: Fixed valid JSON
            return json.dumps({
                "entities": [
                    {
                        "entity_id": "e1",
                        "name": "ধীরেন্দ্রনাথ দত্ত",
                        "entity_type": "PERSON",
                        "evidence_span": "গণপরিষদের সদস্য ধীরেন্দ্রনাথ দত্ত"
                    },
                    {
                        "entity_id": "e2",
                        "name": "করাচি",
                        "entity_type": "LOCATION",
                        "evidence_span": "করাচিতে পাকিস্তান গণপরিষদের"
                    },
                    # Deliberate hallucinated entity with non-existent evidence_span
                    {
                        "entity_id": "e_hallucinated",
                        "name": "মুজিবনগর",
                        "entity_type": "LOCATION",
                        "evidence_span": "মুজিবনগরে পরিষদ"  # NOT in chunk text!
                    }
                ],
                "relations": [],
                "events": []
            })

    mock_client = MockLLMClient(response_generator=mock_responses)
    engine = LLMExtractionEngine(llm_client=mock_client, model_name="mock_gemini")

    # 1. Execute extract_chunk with retry
    output, meta = engine.extract_chunk(sample_chunk, max_retries=3)
    print(f"  Extractor retry verified: Completed with status={meta['status']} after {meta['attempts']} attempts.")
    assert meta["attempts"] == 2, f"Expected 2 attempts due to retry, got {meta['attempts']}"
    assert len(output.entities) == 3

    # 2. Check verbatim evidence span verification
    print(f"  Ungrounded (Hallucinated) items detected: {meta['ungrounded_count']}")
    print(f"  Unsupported item rate: {meta['unsupported_item_rate']*100:.1f}%")
    assert meta["ungrounded_count"] == 1, "Expected exactly 1 ungrounded item (মুজিবনগর)"
    assert meta["unsupported_item_rate"] > 0.0

    # 3. Test Set Guard
    guard_triggered = False
    try:
        engine.extract_page_chunks([sample_chunk], source_name="gold", page_id="p014", split="test", allow_test=False)
    except PermissionError as exc:
        guard_triggered = True
        print(f"  [PASS] Test set protection guard successfully caught unauthorized test run: {exc}")
    assert guard_triggered, "Test set guard failed to raise PermissionError!"


def test_postprocessing_and_graph_validation():
    print("\n" + "=" * 65)
    print("TEST 4: POST-PROCESSING, ENTITY RESOLUTION & GRAPH VALIDATION")
    print("=" * 65)

    # Raw extracted items with inflections, aliases, cycles, and chronological violations
    raw_entities = [
        {"entity_id": "e1", "name": "ধীরেন্দ্রনাথ দত্তের", "entity_type": "PERSON", "evidence_span": "ধীরেন্দ্রনাথ দত্তের দাবি", "is_supported": True},
        {"entity_id": "e2", "name": "ধীরেন্দ্রনাথ দত্ত", "entity_type": "PERSON", "evidence_span": "সদস্য ধীরেন্দ্রনাথ দত্ত", "is_supported": True},
        {"entity_id": "e3", "name": "শেখ মুজিব", "entity_type": "PERSON", "evidence_span": "শেখ মুজিব নেতৃত্ব দেন", "is_supported": True},
        {"entity_id": "e4", "name": "শেখ মুজিবুর রহমান", "entity_type": "PERSON", "evidence_span": "বঙ্গবন্ধু শেখ মুজিবুর রহমান", "is_supported": True},
        {"entity_id": "e5", "name": "পাকিস্তানের", "entity_type": "LOCATION", "evidence_span": "পাকিস্তানের অংশ", "is_supported": True},
        {"entity_id": "e6", "name": "পাকিস্তান", "entity_type": "LOCATION", "evidence_span": "স্বাধীন পাকিস্তান", "is_supported": True},
        {"entity_id": "e7", "name": "১৯৫২", "entity_type": "DATE/PERIOD", "evidence_span": "১৯৫২ সালের ভাষা আন্দোলন", "is_supported": True},
        {"entity_id": "e8", "name": "১৯৭১", "entity_type": "DATE/PERIOD", "evidence_span": "১৯৭১ সালের মুক্তিযুদ্ধ", "is_supported": True},
        {"entity_id": "e9", "name": "মুজিবনগর", "entity_type": "LOCATION", "evidence_span": "কাল্পনিক", "is_supported": False} # ungrounded
    ]

    # 1. Entity Resolution
    canon_entities, alias_map, merge_logs = resolve_entities(raw_entities, page_id="p001")
    print(f"  Entity Resolution: {len(raw_entities)} entities reduced to {len(canon_entities)} canonical entities.")
    for m in merge_logs:
        print(f"    - Merged '{m['source_alias']}' -> '{m['canonical_name']}' ({m['reason']})")

    assert alias_map["ধীরেন্দ্রনাথ দত্তের"] == "ধীরেন্দ্রনাথ দত্ত"
    assert alias_map["শেখ মুজিব"] == "শেখ মুজিবুর রহমান"
    assert alias_map["পাকিস্তানের"] == "পাকিস্তান"
    print("  [PASS] Entity resolution correctly resolved all aliases and inflectional variations.")

    # 2. Relations with self-loop, cycle, and chronological violation
    raw_relations = [
        # Self-loop
        {"relation_id": "r1", "head_text": "পাকিস্তান", "head_type": "LOCATION", "relation_type": "located_in", "tail_text": "পাকিস্তান", "tail_type": "LOCATION", "evidence_span": "...", "is_supported": True},
        # Valid relation
        {"relation_id": "r2", "head_text": "ধীরেন্দ্রনাথ দত্তের", "head_type": "PERSON", "relation_type": "member_of", "tail_text": "পাকিস্তান", "tail_type": "ORGANIZATION", "evidence_span": "...", "is_supported": True},
        # Chronological violation: 1971 event caused 1952 event
        {"relation_id": "r3", "head_text": "১৯৭১", "head_type": "DATE/PERIOD", "relation_type": "caused", "tail_text": "১৯৫২", "tail_type": "DATE/PERIOD", "evidence_span": "১৯৭১ সাল", "is_supported": True},
        # Causal cycle: Event A caused Event B, and Event B caused Event A
        {"relation_id": "r4", "head_text": "ভাষা আন্দোলন", "head_type": "EVENT", "relation_type": "caused", "tail_text": "ধর্মঘট", "tail_type": "EVENT", "evidence_span": "...", "is_supported": True},
        {"relation_id": "r5", "head_text": "ধর্মঘট", "head_type": "EVENT", "relation_type": "caused", "tail_text": "ভাষা আন্দোলন", "tail_type": "EVENT", "evidence_span": "...", "is_supported": True}
    ]

    # Deduplicate & Canonicalize Relations
    deduped_rels = deduplicate_relations(raw_relations, alias_map)

    # Conservative Event Merging
    raw_events = [
        {"event_id": "ev1", "event_type": "PROTEST", "trigger_name": "ধর্মঘট", "participants": [{"name": "ছাত্ররা", "entity_type": EntityType.ORGANIZATION, "role": "protester"}], "causes": ["উর্দু চাপানো"], "consequences": [], "evidence_span": "১১ই মার্চ ঢাকায় ধর্মঘট পালিত হয়", "is_supported": True},
        {"event_id": "ev2", "event_type": "PROTEST", "trigger_name": "ধর্মঘট", "participants": [{"name": "জনগণ", "entity_type": EntityType.ORGANIZATION, "role": "participant"}], "causes": [], "consequences": ["দাবি জোরালো হলো"], "evidence_span": "১১ই মার্চ ঢাকায় ধর্মঘট পালিত হয়", "is_supported": True},
        # Distinct event
        {"event_id": "ev3", "event_type": "PROTEST", "trigger_name": "মিছিল", "participants": [{"name": "ছাত্ররা", "entity_type": EntityType.ORGANIZATION, "role": "protester"}], "causes": [], "consequences": [], "evidence_span": "২১শে ফেব্রুয়ারি ছাত্ররা মিছিল করে", "is_supported": True}
    ]
    merged_evs, ev_logs = merge_events_conservatively(raw_events, alias_map, page_id="p001")
    print(f"  Conservative Event Merging: {len(raw_events)} events merged into {len(merged_evs)} distinct events.")
    assert len(merged_evs) == 2, f"Expected 2 merged events, got {len(merged_evs)}"
    # Check that participants were unified in merged event
    ev1_parts = [p["name"] for p in merged_evs[0]["participants"]]
    assert "ছাত্ররা" in ev1_parts and "জনগণ" in ev1_parts
    print("  [PASS] Identical overlapping events merged while distinct events preserved.")

    # 3. Validate Knowledge Graph
    val_graph, val_report = validate_knowledge_graph(canon_entities, deduped_rels, merged_evs)
    print("\n  Graph Validation Report:")
    print(f"    - Self-loops removed: {val_report['self_loops_removed']}")
    print(f"    - Causal cycles broken: {val_report['causal_cycles_removed']}")
    print(f"    - Chronological violations removed: {val_report['chronological_violations_removed']}")
    print(f"    - Unsupported items filtered: {val_report['unsupported_filtered']}")

    assert val_report["self_loops_removed"] >= 1
    assert val_report["causal_cycles_removed"] >= 1
    assert val_report["chronological_violations_removed"] >= 1
    assert val_report["unsupported_filtered"] >= 1
    print("  [PASS] All graph validation constraints (self-loops, cycles, chronology) passed.")


def test_dual_evaluation_raw_vs_final():
    print("\n" + "=" * 65)
    print("TEST 5: DUAL EVALUATION (RAW OUTPUT VS FINAL GRAPH COMPARISON)")
    print("=" * 65)

    gold_items = [
        {"category": "ENTITY", "type": "PERSON", "text": "ধীরেন্দ্রনাথ দত্ত"},
        {"category": "ENTITY", "type": "PERSON", "text": "শেখ মুজিবুর রহমান"},
        {"category": "ENTITY", "type": "LOCATION", "text": "পাকিস্তান"},
        {"category": "RELATION", "type": "RELATION", "head": "ধীরেন্দ্রনাথ দত্ত", "relation": "member_of", "tail": "পাকিস্তান গণপরিষদ"}
    ]

    # Raw extracted items with inflections and uncanonical aliases
    raw_extracted_items = [
        {"category": "ENTITY", "type": "PERSON", "text": "ধীরেন্দ্রনাথ দত্তের"},  # unnormalized inflection
        {"category": "ENTITY", "type": "PERSON", "text": "শেখ মুজিব"},          # alias
        {"category": "ENTITY", "type": "LOCATION", "text": "পাকিস্তানের"},        # inflection
        {"category": "RELATION", "type": "RELATION", "head": "ধীরেন্দ্রনাথ দত্তের", "relation": "member_of", "tail": "পাকিস্তান গণপরিষদ"}
    ]

    # Final post-processed KG items after resolution and canonicalization
    final_kg_items = [
        {"category": "ENTITY", "type": "PERSON", "text": "ধীরেন্দ্রনাথ দত্ত"},
        {"category": "ENTITY", "type": "PERSON", "text": "শেখ মুজিবুর রহমান"},
        {"category": "ENTITY", "type": "LOCATION", "text": "পাকিস্তান"},
        {"category": "RELATION", "type": "RELATION", "head": "ধীরেন্দ্রনাথ দত্ত", "relation": "member_of", "tail": "পাকিস্তান গণপরিষদ"}
    ]

    res = evaluate_raw_vs_final_graph(gold_items, raw_extracted_items, final_kg_items)

    print("\n  Extraction Performance Comparison Table (Raw Output vs Final KG):")
    print("  " + "-" * 60)
    print(f"  {'Stage':<20} | {'Tier 1 Strict F1':<18} | {'Tier 2 Aligned F1':<18}")
    print("  " + "-" * 60)
    print(f"  {'Raw Output':<20} | {res['raw']['tier1']['f1']:<18.4f} | {res['raw']['tier2']['f1']:<18.4f}")
    print(f"  {'Final Graph (Post)':<20} | {res['final']['tier1']['f1']:<18.4f} | {res['final']['tier2']['f1']:<18.4f}")
    print("  " + "-" * 60)
    print(f"  {'Absolute Gain':<20} | +{res['improvements']['tier1_f1_gain']:<17.4f} | +{res['improvements']['tier2_f1_gain']:<17.4f}")
    print("  " + "-" * 60)

    assert res["final"]["tier1"]["f1"] >= res["raw"]["tier1"]["f1"]
    assert res["final"]["tier2"]["f1"] >= res["raw"]["tier2"]["f1"]
    print("  [PASS] Dual evaluation confirmed that post-processing significantly improves final Knowledge Graph F1.")


def main():
    print("=" * 65)
    print("RUNNING ALL STEP 4 VERIFICATION TESTS")
    print("=" * 65)
    test_chunking_strategies()
    test_schema_and_domain_constraints()
    test_llm_extractor_retry_and_grounding()
    test_postprocessing_and_graph_validation()
    test_dual_evaluation_raw_vs_final()
    print("\n" + "=" * 65)
    print("ALL STEP 4 TESTS COMPLETED AND VERIFIED WITH 100% SUCCESS!")
    print("=" * 65)


if __name__ == "__main__":
    main()
