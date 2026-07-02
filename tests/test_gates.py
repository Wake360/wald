"""Golden gates G0 + G1 (need a built corpus: `wald corpus build`)."""

import json
from pathlib import Path

import pytest

from wald.eval import evaluate

CORPUS = Path(__file__).parent.parent / "corpus"

pytestmark = pytest.mark.skipif(
    not (CORPUS / "MANIFEST.json").exists(),
    reason="corpus not built (run: wald corpus build)",
)


def manifest():
    return json.loads((CORPUS / "MANIFEST.json").read_text())


def test_g0_corpus_size_and_verification():
    m = manifest()
    assert len(m["clean"]) >= 20
    assert len(m["mutants"]) >= 60
    assert all(e["verified"] for e in m["mutants"])
    for e in m["mutants"]:
        assert (CORPUS / e["file"]).exists()
    assert "provenance" in m


def test_g0_every_flaw_class_with_mutation_is_represented():
    labels = {e["flaw_id"] for e in manifest()["mutants"]}
    assert {
        "leakage-fit-before-split",
        "testing-multiple-uncorrected",
        "baserate-accuracy-imbalanced",
        "selection-survivorship-cohort",
    } <= labels


def test_g1_static_precision_recall_and_clean_fp():
    results = evaluate(CORPUS)
    for cls, r in results["static_classes"].items():
        assert r["precision"] is None or r["precision"] >= 0.9, (cls, r)
        assert r["recall"] is not None and r["recall"] >= 0.7, (cls, r)
    assert results["clean_fp_rate"] <= 0.05, results["clean_fp_files"]


def test_g1_survivorship_candidate_recall():
    results = evaluate(CORPUS)
    r = results["candidate_classes"]["selection-survivorship-cohort"]
    assert r["recall"] is not None and r["recall"] >= 0.8, r
