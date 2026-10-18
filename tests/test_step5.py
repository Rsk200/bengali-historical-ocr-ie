#!/usr/bin/env python3
"""
Comprehensive Unit Test and Verification Suite for Step 5:
  1. Label Studio Setup, Gold Converter, and Inter-Annotator Agreement (IAA) Module
  2. Post-OCR Correction Ablation Module (Arms A, B, C with Name/Number Guard & Alignment Trimming)
  3. Master Experiment & Ablation Orchestrator (C2 Selection, C3 Error Propagation, CER-Binning & Predictors, A1-A3 Ablations)
  4. Google Colab GPU Runner Notebook Structure Verification
  5. Test Set Read-Only Guard & Code Freeze / Tagging Protocol Verification
"""

import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.annotation_utils import (
    cohen_kappa_on_aligned_spans,
    compute_iaa_f1,
    convert_label_studio_to_gold,
    get_annotation_schedule_week,
    prepare_label_studio_tasks
)
from src.post_ocr_correction import (
    CORE_BANGLA_DICTIONARY,
    PROTECTED_HISTORICAL_NAMES,
    correct_arm_a_none,
    correct_arm_b_contextual_bert_dict,
    correct_arm_c_llm_fix_only,
    evaluate_correction_arm,
    is_protected_token,
    trim_overgeneration_by_alignment
)
from src.run_experiments import (
    analyze_error_propagation_bins_and_predictors,
    freeze_and_tag_for_final_test,
    run_a1_correction_ablation,
    run_a2_chunking_ablation,
    run_a3_prompt_model_consistency_ablation,
    run_c2_ocr_selection,
    run_c3_error_propagation
)


# ==============================================================================
# TEST 1: Label Studio Setup, Schedule, Gold Converter & IAA
# ==============================================================================

def test_annotation_utils_and_iaa():
    print("\n" + "=" * 65)
    print("TEST 1: LABEL STUDIO SETUP, GOLD CONVERTER & IAA MODULE")
    print("=" * 65)

    # 1.1 Verify Label Studio XML Interface
    xml_path = Path("configs/label_studio_interface.xml")
    assert xml_path.exists(), "configs/label_studio_interface.xml does not exist!"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")
    
    # Required Entity and Relation Types
    required_entities = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "DATE/PERIOD", "DOCUMENT/TREATY/LAW"]
    for ent in required_entities:
        assert ent in xml_str, f"Missing required entity label '{ent}' in label_studio_interface.xml"
    
    required_relations = ["participated_in", "led_by", "member_of", "occurred_at", "occurred_on", "caused", "resulted_in"]
    for rel in required_relations:
        assert rel in xml_str, f"Missing required relation label '{rel}' in label_studio_interface.xml"
    print("  [Pass] Label Studio XML interface contains all required entity and relation types.")

    # 1.2 Verify 4-Week Schedule & Task Preparation
    tasks = prepare_label_studio_tasks(
        pages_csv_path="data/pages.csv",
        output_json_path="data/gold_standard/label_studio_tasks.json"
    )
    assert len(tasks) == 38, f"Expected exactly 38 core prose tasks, got {len(tasks)}"

    week_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    split_counts = {"dev": 0, "test": 0}
    for t in tasks:
        w = t["data"]["schedule_week"]
        s = t["data"]["split"]
        week_counts[w] += 1
        split_counts[s] += 1

    assert week_counts[1] == 5, f"Week 1 Pilot Calibration expected 5 pages, got {week_counts[1]}"
    assert week_counts[2] == 6, f"Week 2 Dev Remaining expected 6 pages, got {week_counts[2]}"
    assert week_counts[3] == 11, f"Week 3 Test Part 1 expected 11 pages, got {week_counts[3]}"
    assert week_counts[4] == 16, f"Week 4 Test Part 2 expected 16 pages, got {week_counts[4]}"
    assert split_counts["dev"] == 11, f"Expected 11 Dev pages, got {split_counts['dev']}"
    assert split_counts["test"] == 27, f"Expected 27 Test pages, got {split_counts['test']}"
    print(f"  [Pass] 4-Week Schedule verified: Week 1={week_counts[1]}, Week 2={week_counts[2]}, "
          f"Week 3={week_counts[3]}, Week 4={week_counts[4]} (Dev: {split_counts['dev']}, Test: {split_counts['test']})")

    # 1.3 Verify Gold Converter with Character Offsets
    sample_gold_text = (
        "১৯৪৮ সালের ২৩শে ফেব্রুয়ারি করাচিতে পাকিস্তান গণপরিষদের প্রথম অধিবেশন শুরু হয়। "
        "ধীরেন্দ্রনাথ দত্ত বাংলা ব্যবহারের দাবি জানান।"
    )
    s_dhiren = sample_gold_text.find("ধীরেন্দ্রনাথ দত্ত")
    e_dhiren = s_dhiren + len("ধীরেন্দ্রনাথ দত্ত")
    s_karachi = sample_gold_text.find("করাচিতে")
    e_karachi = s_karachi + len("করাচিতে")

    mock_ls_task = [{
        "id": "p001_test",
        "data": {
            "page_id": "p001",
            "chapter_id": "ch01",
            "split": "dev",
            "text": sample_gold_text
        },
        "annotations": [{
            "result": [
                {
                    "id": "ent_1",
                    "type": "labels",
                    "value": {"start": s_dhiren, "end": e_dhiren, "text": "ধীরেন্দ্রনাথ দত্ত", "labels": ["PERSON"]}
                },
                {
                    "id": "ent_2",
                    "type": "labels",
                    "value": {"start": s_karachi, "end": e_karachi, "text": "করাচিতে", "labels": ["LOCATION"]}
                },
                {
                    "id": "rel_1",
                    "type": "relation",
                    "from_id": "ent_1",
                    "to_id": "ent_2",
                    "labels": ["participated_in"]
                }
            ]
        }]
    }]

    converted_files = convert_label_studio_to_gold(mock_ls_task, output_base_dir="data/gold_standard")
    assert len(converted_files) == 1
    gold_file = converted_files[0]
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_record = json.load(f)

    assert gold_record["page_id"] == "p001"
    assert len(gold_record["entities"]) == 2
    assert len(gold_record["relations"]) == 1
    # Check exact slice match
    for ent in gold_record["entities"]:
        slice_text = gold_record["gold_text"][ent["char_start"]:ent["char_end"]]
        assert slice_text == ent["name"], f"Slice mismatch: '{slice_text}' != '{ent['name']}'"
    print("  [Pass] Label Studio to Gold Converter: exact character offsets verified.")

    # 1.4 Verify Inter-Annotator Agreement (IAA) F1
    annotator_1 = {
        "entities": [
            {"entity_type": "PERSON", "char_start": 0, "char_end": 16, "name": "ধীরেন্দ্রনাথ দত্ত"},
            {"entity_type": "LOCATION", "char_start": 20, "char_end": 27, "name": "করাচিতে"}
        ],
        "relations": [
            {"head_text": "ধীরেন্দ্রনাথ দত্ত", "relation_type": "participated_in", "tail_text": "গণপরিষদ"}
        ],
        "events": [{
            "trigger_name": "অধিবেশন",
            "event_type": "POLITICAL_SESSION",
            "date": "২৩শে ফেব্রুয়ারি ১৯৪৮",
            "location": "করাচি",
            "causes": ["পাকিস্তান রাষ্ট্রের সূচনা"],
            "consequences": ["বাংলা ভাষার দাবি উত্থাপন"]
        }]
    }

    # Annotator 2: identical annotations -> F1 should be 1.0
    iaa_perfect = compute_iaa_f1(annotator_1, annotator_1)
    assert iaa_perfect["entity_exact_f1"] == 1.0
    assert iaa_perfect["relation_tuple_f1"] == 1.0
    assert iaa_perfect["event_atomic_slot_f1"] == 1.0
    assert iaa_perfect["overall_iaa_f1"] == 1.0
    print("  [Pass] IAA Perfect Agreement verified (F1 = 1.0000).")

    # Annotator 2 with minor inflection differences & descriptive overlap
    annotator_2 = {
        "entities": [
            {"entity_type": "PERSON", "char_start": 0, "char_end": 18, "name": "ধীরেন্দ্রনাথ দত্তের"},  # relaxed match (inflected)
            {"entity_type": "LOCATION", "char_start": 20, "char_end": 27, "name": "করাচি"}
        ],
        "relations": [
            {"head_text": "ধীরেন্দ্রনাথ দত্তের", "relation_type": "participated_in", "tail_text": "গণপরিষদে"}  # stem match
        ],
        "events": [{
            "trigger_name": "অধিবেশন",
            "event_type": "POLITICAL_SESSION",
            "date": "২৩শে ফেব্রুয়ারি ১৯৪৮",
            "location": "করাচি",
            "causes": ["পাকিস্তান রাষ্ট্রের সূচনা ও পটভূমি"],  # token overlap >= 0.5
            "consequences": ["বাংলা ভাষার ঐতিহাসিক দাবি উত্থাপন"]
        }]
    }
    iaa_eval = compute_iaa_f1(annotator_1, annotator_2)
    assert iaa_eval["entity_relaxed_f1"] >= 0.8
    assert iaa_eval["relation_tuple_f1"] == 1.0
    assert iaa_eval["event_atomic_slot_f1"] == 1.0
    assert iaa_eval["event_descriptive_slot_f1"] == 1.0
    print(f"  [Pass] IAA Realistic Agreement: Entity Relaxed={iaa_eval['entity_relaxed_f1']}, "
          f"Relation={iaa_eval['relation_tuple_f1']}, Event Slots={iaa_eval['event_atomic_slot_f1']}")

    # 1.5 Verify Cohen's Kappa on Aligned Spans
    spans_a1 = [{"entity_type": "PERSON", "char_start": 0, "char_end": 16}, {"entity_type": "LOCATION", "char_start": 20, "char_end": 27}]
    spans_a2 = [{"entity_type": "PERSON", "char_start": 0, "char_end": 18}, {"entity_type": "LOCATION", "char_start": 20, "char_end": 26}]
    kappa_res = cohen_kappa_on_aligned_spans(spans_a1, spans_a2, min_iou=0.5)
    assert kappa_res["observed_agreement"] == 1.0
    assert kappa_res["cohen_kappa"] == 1.0
    assert kappa_res["n_aligned_spans"] == 2
    print(f"  [Pass] Cohen's Kappa on Aligned Spans: kappa={kappa_res['cohen_kappa']}, N={kappa_res['n_aligned_spans']}")


# ==============================================================================
# TEST 2: Post-OCR Correction Ablation Module (Arms A, B, C)
# ==============================================================================

def test_post_ocr_correction():
    print("\n" + "=" * 65)
    print("TEST 2: POST-OCR CORRECTION ARMS (A, B, C) & SAFEGUARDS")
    print("=" * 65)

    raw_ocr = (
        "১৯৪৮ সালের ২৩শে ফেব্রুয়ারি করাচিতে পাকিস্তান গণপরিষদের প্রথম অধিবেশন শুরু হয়। "
        "ধীরেন্দ্রনাথ দত্ত পরিষদের ভাষা হিসেবে বাংলা ব্যবহারের ঐতহাসিক দাবি জানান।"
    )
    gold_text = (
        "১৯৪৮ সালের ২৩শে ফেব্রুয়ারি করাচিতে পাকিস্তান গণপরিষদের প্রথম অধিবেশন শুরু হয়। "
        "ধীরেন্দ্রনাথ দত্ত পরিষদের ভাষা হিসেবে বাংলা ব্যবহারের ঐতিহাসিক দাবি জানান।"
    )

    # 2.1 Arm A: None (Baseline)
    arm_a_out = correct_arm_a_none(raw_ocr)
    assert arm_a_out == raw_ocr, "Arm A must return identical raw OCR text!"
    res_a = evaluate_correction_arm("Arm A", raw_ocr, arm_a_out, gold_text)
    assert res_a["cer_reduction"] == 0.0
    assert res_a["change_ratio"] == 0.0
    print(f"  [Pass] Arm A (Baseline): raw_cer={res_a['raw_cer']}, cer_reduction={res_a['cer_reduction']}")

    # 2.2 Arm B: Contextual Spell-Checker with Name & Number Guard
    # Check that protected entities and dates are NEVER modified
    assert is_protected_token("ধীরেন্দ্রনাথ"), "Protected historical name stem must be protected!"
    assert is_protected_token("১৯৪৮"), "Bengali 4-digit date must be protected!"
    assert is_protected_token("২৩শে"), "Bengali date with suffix must be protected!"

    # In raw_ocr, "ঐতহাসিক" is corrupted (missing i-kar \u09bf). In CORE_BANGLA_DICTIONARY, "ঐতিহাসিক" is present.
    arm_b_out = correct_arm_b_contextual_bert_dict(raw_ocr)
    assert "ধীরেন্দ্রনাথ দত্ত" in arm_b_out, "Name guard failed: historical person altered!"
    assert "১৯৪৮" in arm_b_out and "২৩শে" in arm_b_out, "Number guard failed: historical date altered!"
    assert "ঐতিহাসিক" in arm_b_out, "Spell-check failed: 'ঐতহাসিক' was not corrected to 'ঐতিহাসিক'!"

    res_b = evaluate_correction_arm("Arm B", raw_ocr, arm_b_out, gold_text)
    assert res_b["cer_reduction"] > 0, "Arm B should reduce CER on dictionary typos!"
    print(f"  [Pass] Arm B (Contextual Dict + Guard): raw_cer={res_b['raw_cer']}, post_cer={res_b['post_cer']}, "
          f"CER reduction={res_b['cer_reduction']}, Name & Number Guard active.")

    # 2.3 Arm C: LLM Fix Only with Alignment Trimming & Number Restoration
    hallucinated_llm_output = (
        "এখানে সংশোধিত অনুচ্ছেদ হলো:\n"
        "১৯৫০ সালের ২৩শে ফেব্রুয়ারি করাচিতে পাকিস্তান গণপরিষদের প্রথম অধিবেশন শুরু হয়। "
        "ধীরেন্দ্রনাথ দত্ত পরিষদের ভাষা হিসেবে বাংলা ব্যবহারের ঐতিহাসিক দাবি জানান। "
        "এটি ছিল পূর্ব বাংলার স্বাধিকার আন্দোলনের এক গৌরবোজ্জ্বল অধ্যায়।"
    )
    # Notice: LLM silently changed 1948 to 1950, added intro conversational prefix, and hallucinated an extra sentence.
    trimmed_c = trim_overgeneration_by_alignment(raw_ocr, hallucinated_llm_output, max_length_ratio=1.15)

    # Verify conversational prefix stripped
    assert not trimmed_c.startswith("এখানে সংশোধিত অনুচ্ছেদ হলো:"), "Conversational prefix was not stripped!"
    # Verify original year restored
    assert "১৯৪৮" in trimmed_c, "Safeguard failed: original year '১৯৪৮' was not restored from OCR!"
    # Verify extra hallucinated sentence trimmed
    assert "গৌরবোজ্জ্বল অধ্যায়" not in trimmed_c, "Alignment trimming failed to eliminate hallucinated addition!"
    print("  [Pass] Arm C Alignment Trimming: prefix stripped, hallucination pruned, original date restored.")


# ==============================================================================
# TEST 3: Master Experiment Runner (C2, C3, Predictors, A1-A3, Freeze)
# ==============================================================================

def test_master_experiments():
    print("\n" + "=" * 65)
    print("TEST 3: MASTER EXPERIMENTS & ABLATION RUNNER")
    print("=" * 65)

    # 3.1 C2: OCR System Selection on Dev Set
    mock_ocr_results = {
        "easyocr": [
            {"page_id": "p001", "normalised_text": "১৯৪৮ সালের ২৩শে ফেব্রুয়ারি করাচিতে পাকিস্তান গণপরিষদের প্রথম অধিবেশন শুরু হয়। ধীরেন্দ্রনাথ দত্ত বাংলা ব্যবহারের দাবি জানান।"},
            {"page_id": "p002", "normalised_text": "১৯৫২ সালের ২১শে ফেব্রুয়ারি ছাত্ররা ১৪৪ ধারা ভঙ্গ করে মিছল বের করে।"}
        ],
        "tesseract": [
            {"page_id": "p001", "normalised_text": "১৯৪৮ সালের করাচিতে পাকিস্তান অধিবেশন শুরু হয় ধীরেন দত্ত দাবি জানান"},
            {"page_id": "p002", "normalised_text": "১৯৫২ ২১শে ফেব্রুয়ারি ১৪৪ ধারা ভঙ্গ মিছিল করে"}
        ]
    }
    mock_gold = {
        "p001": {
            "gold_text": "১৯৪৮ সালের ২৩শে ফেব্রুয়ারি করাচিতে পাকিস্তান গণপরিষদের প্রথম অধিবেশন শুরু হয়। ধীরেন্দ্রনাথ দত্ত পরিষদের ভাষা হিসেবে বাংলা ব্যবহারের দাবি জানান।",
            "entities": [{"name": "ধীরেন্দ্রনাথ দত্ত", "entity_type": "PERSON"}, {"name": "করাচি", "entity_type": "LOCATION"}]
        },
        "p002": {
            "gold_text": "১৯৫২ সালের ২১শে ফেব্রুয়ারি ছাত্ররা ১৪৪ ধারা ভঙ্গ করে ঐতিহাসিক মিছিল বের করে।",
            "entities": [{"name": "২১শে ফেব্রুয়ারি", "entity_type": "DATE/PERIOD"}, {"name": "ছাত্ররা", "entity_type": "ORGANIZATION"}]
        }
    }

    c2_res = run_c2_ocr_selection(mock_ocr_results, mock_gold, split="dev")
    assert c2_res["selected_strong_ocr"] == "easyocr", f"Expected easyocr as strong OCR, got {c2_res['selected_strong_ocr']}"
    assert c2_res["selected_weaker_ocr"] == "tesseract", f"Expected tesseract as weaker OCR, got {c2_res['selected_weaker_ocr']}"
    print(f"  [Pass] C2 Selection: Strong OCR='{c2_res['selected_strong_ocr']}', Weaker OCR='{c2_res['selected_weaker_ocr']}'")

    # 3.2 C3: Error Propagation & Three-Part Error Attribution
    mock_c3_systems = {
        "oracle": {
            "p001": {
                "text": mock_gold["p001"]["gold_text"],
                "extracted_items": [{"category": "ENTITY", "type": "PERSON", "text": "ধীরেন্দ্রনাথ দত্ত", "char_start": 80, "char_end": 96}]
            }
        },
        "strong_ocr": {
            "p001": {
                "text": mock_ocr_results["easyocr"][0]["normalised_text"],
                "extracted_items": [{"category": "ENTITY", "type": "PERSON", "text": "ধীরেন্দ্রনাথ দত্ত", "char_start": 80, "char_end": 96}]
            }
        },
        "weaker_ocr": {
            "p001": {
                "text": mock_ocr_results["tesseract"][0]["normalised_text"],
                "extracted_items": [{"category": "ENTITY", "type": "PERSON", "text": "ধীরেন দত্ত", "char_start": 40, "char_end": 50}]
            }
        }
    }

    c3_res = run_c3_error_propagation(mock_gold, mock_c3_systems, split="dev", allow_test=False)
    assert "oracle" in c3_res["condition_performances"]
    assert "strong_ocr" in c3_res["condition_performances"]
    assert "weaker_ocr" in c3_res["condition_performances"]

    for cond, perf in c3_res["condition_performances"].items():
        assert perf["tier2_f1"] >= perf["tier1_f1"], f"Tier 2 F1 ({perf['tier2_f1']}) must be >= Tier 1 F1 ({perf['tier1_f1']}) for {cond}!"

    print(f"  [Pass] C3 Error Propagation: Tier 2 >= Tier 1 verified across all conditions.")

    # 3.3 Test Set Read-Only Guard Verification
    # Attempting to run on 'test' split without allow_test=True MUST raise PermissionError
    test_guard_passed = False
    try:
        run_c3_error_propagation(mock_gold, mock_c3_systems, split="test", allow_test=False)
    except PermissionError:
        test_guard_passed = True
    assert test_guard_passed, "Security Guard Failed: C3 allowed execution on 'test' split without permission!"

    # Ablations A1, A2, A3 must also raise PermissionError on split='test'
    for fn, args in [
        (run_a1_correction_ablation, ([{"ocr_text": "a", "gold_text": "a"}], "test")),
        (run_a2_chunking_ablation, ("sample", "p001", "test")),
        (run_a3_prompt_model_consistency_ablation, ({}, None, "test"))
    ]:
        try:
            fn(*args)
            assert False, f"{fn.__name__} failed to raise PermissionError on split='test'!"
        except PermissionError:
            pass
    print("  [Pass] Test Set Read-Only Guards strictly verified (PermissionError raised on all ablations).")

    # 3.4 CER-Binning & Correlation Predictor Analysis
    page_stats = [
        {"page_id": "p1", "cer": 0.012, "target_accuracy": 0.98, "f1_drop": 0.02},
        {"page_id": "p2", "cer": 0.018, "target_accuracy": 0.96, "f1_drop": 0.03},
        {"page_id": "p3", "cer": 0.035, "target_accuracy": 0.91, "f1_drop": 0.08},
        {"page_id": "p4", "cer": 0.042, "target_accuracy": 0.88, "f1_drop": 0.12},
        {"page_id": "p5", "cer": 0.065, "target_accuracy": 0.72, "f1_drop": 0.25},
        {"page_id": "p6", "cer": 0.082, "target_accuracy": 0.65, "f1_drop": 0.32}
    ]
    bin_analysis = analyze_error_propagation_bins_and_predictors(page_stats)
    assert bin_analysis["cer_bins"]["low_cer_under_2pct"]["n_pages"] == 2
    assert bin_analysis["cer_bins"]["moderate_cer_2_to_5pct"]["n_pages"] == 2
    assert bin_analysis["cer_bins"]["high_cer_over_5pct"]["n_pages"] == 2

    # Verify correlations are calculated and meaningful
    corr_cer = bin_analysis["predictor_analysis"]["cer_vs_f1_drop_correlation"]
    corr_target = bin_analysis["predictor_analysis"]["target_accuracy_vs_f1_drop_correlation"]
    assert corr_cer > 0.8, f"Expected strong positive correlation for CER vs F1 drop, got {corr_cer}"
    assert corr_target < -0.8, f"Expected strong negative correlation for Target Accuracy vs F1 drop, got {corr_target}"
    print(f"  [Pass] CER-Binning & Predictor Analysis: Low Drop={bin_analysis['cer_bins']['low_cer_under_2pct']['mean_f1_drop']}, "
          f"High Drop={bin_analysis['cer_bins']['high_cer_over_5pct']['mean_f1_drop']}, "
          f"r(CER)={corr_cer}, r(TargetAcc)={corr_target}, Stronger={bin_analysis['predictor_analysis']['stronger_predictor']}")

    # 3.5 Dev Ablations A1, A2, A3
    a1_res = run_a1_correction_ablation([{"ocr_text": "ঐতহাসিক আন্দোলন", "gold_text": "ঐতিহাসিক আন্দোলন"}], split="dev")
    assert "mean_cer_reduction" in a1_res

    a2_res = run_a2_chunking_ablation("প্রথম অধ্যায়\n\nপরিচ্ছেদ ১.১\n\n১৯৪৮ সালের ২৩শে ফেব্রুয়ারি।", page_id="p001", split="dev")
    assert a2_res["chunk_counts"]["S1_structure_aware"] >= 1

    a3_res = run_a3_prompt_model_consistency_ablation({}, split="dev", n_runs=3, temperature=0.2)
    assert a3_res["pairwise_f1_agreement_mean"] == 1.0
    assert a3_res["pairwise_f1_agreement_std"] == 0.0
    print("  [Pass] Dev Ablations A1, A2, A3 verified successfully.")

    # 3.6 Code Freeze Protocol Verification
    freeze_res = freeze_and_tag_for_final_test(tag_name="v1.0-pre-test-verify-test")
    assert freeze_res["checks"]["config_seed_42"] is True, "Config seed check failed!"
    assert freeze_res["checks"]["requirements_pinned"] is True, "Requirements pinned check failed!"
    print(f"  [Pass] Code Freeze Protocol: seed=42 verified, requirements pinned verified.")


# ==============================================================================
# TEST 4: Google Colab GPU Runner Notebook Verification
# ==============================================================================

def test_colab_gpu_notebook():
    print("\n" + "=" * 65)
    print("TEST 4: GOOGLE COLAB GPU RUNNER NOTEBOOK VERIFICATION")
    print("=" * 65)

    nb_path = Path("notebooks/colab_gpu_runner.ipynb")
    assert nb_path.exists(), "notebooks/colab_gpu_runner.ipynb does not exist!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    assert nb_json.get("nbformat") == 4, f"Expected nbformat 4, got {nb_json.get('nbformat')}"
    cells = nb_json.get("cells", [])
    assert len(cells) >= 8, f"Expected at least 8 notebook cells, got {len(cells)}"

    # Check key GPU components in notebook
    full_nb_text = json.dumps(nb_json, ensure_ascii=False)
    required_keywords = [
        "nvidia-smi",
        "easyocr",
        "sagorsarker/banglabert",
        "Qwen/Qwen2.5-7B-Instruct",
        "colab_export.zip"
    ]
    for kw in required_keywords:
        assert kw in full_nb_text, f"Missing required keyword '{kw}' in Colab runner notebook!"

    print("  [Pass] Colab GPU runner notebook valid JSON, nbformat 4, contains all required OCR/LLM pipelines.")


# ==============================================================================
# MAIN TEST EXECUTION
# ==============================================================================

def main():
    print("=" * 65)
    print("STARTING STEP 5 COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 65)

    test_annotation_utils_and_iaa()
    test_post_ocr_correction()
    test_master_experiments()
    test_colab_gpu_notebook()

    print("\n" + "=" * 65)
    print("ALL STEP 5 VERIFICATION TESTS PASSED SUCCESSFULLY (100%)!")
    print("=" * 65)


if __name__ == "__main__":
    main()
