"""Corpus eval: detectors vs. labeled mutants -> confusion matrix per class.

This is the project's evidence. `wald eval` runs the static layer over the
whole corpus and writes a dated report; gates G0/G1 assert on its output.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .detect import DEFAULT_CONFIDENCE_FLOOR, STATIC_DECIDABLE, run_static
from .ingest import parse_notebook


def evaluate(corpus_root: str | Path, floor: float = DEFAULT_CONFIDENCE_FLOOR) -> dict:
    root = Path(corpus_root)
    manifest = json.loads((root / "MANIFEST.json").read_text())

    per_class = {c: {"tp": 0, "fn": 0, "fp": 0} for c in sorted(STATIC_DECIDABLE)}
    candidate = {"selection-survivorship-cohort": {"tp": 0, "fn": 0}}
    clean_fp_files = []
    misses = []

    for entry in manifest["clean"]:
        flags = run_static(parse_notebook(root / entry["file"]))
        confident = [f for f in flags if f.confidence >= floor and f.flaw_id in STATIC_DECIDABLE]
        for f in confident:
            per_class[f.flaw_id]["fp"] += 1
        if confident:
            clean_fp_files.append(entry["file"])

    for entry in manifest["mutants"]:
        label = entry["flaw_id"]
        flags = run_static(parse_notebook(root / entry["file"]))
        if label in STATIC_DECIDABLE:
            hit = any(f.flaw_id == label and f.confidence >= floor for f in flags)
            per_class[label]["tp" if hit else "fn"] += 1
            if not hit:
                misses.append(entry["file"])
        elif label in candidate:
            hit = any(f.flaw_id == label for f in flags)  # any confidence: candidate layer
            candidate[label]["tp" if hit else "fn"] += 1
            if not hit:
                misses.append(entry["file"])
        # spurious confident flags of OTHER static classes on a mutant = FP
        for f in flags:
            if f.flaw_id in STATIC_DECIDABLE and f.flaw_id != label and f.confidence >= floor:
                per_class[f.flaw_id]["fp"] += 1

    def prf(c):
        tp, fn, fp = c["tp"], c["fn"], c["fp"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        return {"tp": tp, "fn": fn, "fp": fp, "precision": precision, "recall": recall}

    n_clean = len(manifest["clean"])
    results = {
        "date": date.today().isoformat(),
        "corpus_built": manifest["built"],
        "n_clean": n_clean,
        "n_mutants": len(manifest["mutants"]),
        "n_discarded": len(manifest["discarded"]),
        "confidence_floor": floor,
        "static_classes": {c: prf(v) for c, v in per_class.items()},
        "candidate_classes": {
            c: {**v, "recall": v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else None}
            for c, v in candidate.items()
        },
        "clean_fp_rate": len(clean_fp_files) / n_clean if n_clean else None,
        "clean_fp_files": clean_fp_files,
        "missed_mutants": misses,
    }
    return results


def render_report(results: dict) -> str:
    lines = [
        f"# Wald eval — {results['date']} (corpus built {results['corpus_built']})",
        "",
        f"{results['n_clean']} clean notebooks, {results['n_mutants']} verified mutants "
        f"({results['n_discarded']} discarded at build), confidence floor "
        f"{results['confidence_floor']}.",
        "",
        "## Static classes (layer A, decides alone)",
        "",
        "| class | TP | FN | FP | precision | recall |",
        "|---|---|---|---|---|---|",
    ]
    for c, r in results["static_classes"].items():
        p = f"{r['precision']:.2f}" if r["precision"] is not None else "—"
        rc = f"{r['recall']:.2f}" if r["recall"] is not None else "—"
        lines.append(f"| {c} | {r['tp']} | {r['fn']} | {r['fp']} | {p} | {rc} |")
    lines += [
        "",
        f"False-positive rate on clean corpus: "
        f"{results['clean_fp_rate']:.1%} ({len(results['clean_fp_files'])} files)",
        "",
        "## Candidate classes (static half only; fusion with narrative layer is M2)",
        "",
    ]
    for c, r in results["candidate_classes"].items():
        rc = f"{r['recall']:.2f}" if r["recall"] is not None else "—"
        lines.append(f"- {c}: candidate recall {rc} ({r['tp']}/{r['tp'] + r['fn']})")
    if results["missed_mutants"]:
        lines += ["", "## Missed mutants"]
        lines += [f"- {m}" for m in results["missed_mutants"]]
    lines += [
        "",
        "## Honest caveats",
        "- The corpus is synthetic and stereotypical by design (v1); these "
        "numbers measure detector correctness on canonical idioms, not "
        "real-world recall. Dogfooding on real notebooks is milestone M4.",
        "- Survivorship is reported as candidate recall only — the static "
        "half cannot decide it (the flaw is the pair filter+claim).",
    ]
    return "\n".join(lines)


def run_eval(corpus_root: str | Path, out_dir: str | Path = "evals") -> dict:
    results = evaluate(corpus_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{results['date']}-eval.json").write_text(json.dumps(results, indent=2))
    (out / f"{results['date']}-eval.md").write_text(render_report(results))
    return results
