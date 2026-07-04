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
from .llm import PINNED_DETECTOR_MODEL, PINNED_VERIFIER_MODEL
from .taxonomy import load_taxonomy


def evaluate(corpus_root: str | Path, floor: float = DEFAULT_CONFIDENCE_FLOOR,
             flags_producer=run_static) -> dict:
    root = Path(corpus_root)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    # reviewed real notebooks (corpus/real) join the clean set; wald corpus
    # build regenerates only the synthetic MANIFEST, so they live separately
    real_manifest = root / "real" / "MANIFEST.json"
    n_real = 0
    if real_manifest.exists():
        real_clean = json.loads(real_manifest.read_text())["clean"]
        n_real = len(real_clean)
        manifest["clean"] = manifest["clean"] + real_clean

    per_class = {c: {"tp": 0, "fn": 0, "fp": 0} for c in sorted(STATIC_DECIDABLE)}
    candidate = {"selection-survivorship-cohort": {"tp": 0, "fn": 0}}
    clean_fp_files = []
    misses = []

    for entry in manifest["clean"]:
        flags = flags_producer(parse_notebook(root / entry["file"]))
        confident = [f for f in flags if f.confidence >= floor and f.flaw_id in STATIC_DECIDABLE]
        for f in confident:
            per_class[f.flaw_id]["fp"] += 1
        if confident:
            clean_fp_files.append(entry["file"])

    for entry in manifest["mutants"]:
        label = entry["flaw_id"]
        flags = flags_producer(parse_notebook(root / entry["file"]))
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
        "n_clean_real": n_real,
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
        f"{results['n_clean']} clean notebooks "
        f"({results.get('n_clean_real', 0)} real, reviewed; rest synthetic), "
        f"{results['n_mutants']} verified mutants "
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
        "- Mutants are injected only into the synthetic notebooks, so recall "
        "is measured on canonical idioms. Real notebooks contribute to the "
        "clean FP rate only; real-flaw recall rests on the 7 confirmed "
        "instances in the dogfood report (too few for a recall number).",
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


REAL_FP_CAVEAT = (
    "corpus/real shaped the static layer during dogfood, so the fused-FP "
    "number leans optimistic — it is still the only real-notebook FP "
    "evidence available"
)


def _prf1(c: dict) -> dict:
    tp, fn, fp = c["tp"], c["fn"], c["fp"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fn": fn, "fp": fp, "precision": precision, "recall": recall, "f1": f1}


def evaluate_narrative(corpus_root: str | Path, det_backend, ver_backend,
                       split: str = "dev", floor: float = DEFAULT_CONFIDENCE_FLOOR) -> dict:
    from .fuse import _VerifyTarget, run_full_traced
    from .verifier import verify_finding

    # structural guard (risk R3): held-out data may only ever meet fresh API
    # backends — contamination is prevented, not labeled. `kind` is static
    # (set at construction), unlike `gate_eligible`, which a pre-populated
    # ReplayBackend would satisfy at check time and then violate mid-run by
    # serving cached responses from disk.
    for role, b in (("detector", det_backend), ("verifier", ver_backend)):
        if split == "heldout" and b.kind != "api":
            raise RuntimeError(
                f"heldout split refused: {role} backend (kind={b.kind!r}) "
                "is not gate-eligible"
            )

    root = Path(corpus_root)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    per_class = {fid: {"tp": 0, "fn": 0, "fp": 0}
                 for fid, d in sorted(load_taxonomy().items()) if d.narrative_enabled}
    n_dropped = n_raw_findings = 0
    survival = {"supported": 0, "total": 0}
    clean_fp_files = []
    misses = []

    def narrative_flags(file):
        nonlocal n_dropped, n_raw_findings
        survivors, narrative, fused = run_full_traced(
            parse_notebook(root / file), det_backend, ver_backend
        )
        dropped_here = sum(1 for d in narrative.dropped if "finding dropped" in d)
        n_dropped += dropped_here
        n_raw_findings += len(narrative.findings) + dropped_here
        derived = [f for f in survivors if f.extra.get("narrative_derived")]
        kept = {id(f) for f in derived}
        return derived, fused, kept

    for entry in manifest["mutants"]:
        if entry["split"] != split or entry["flaw_id"] not in per_class:
            continue
        label = entry["flaw_id"]
        flags, fused, kept = narrative_flags(entry["file"])
        hit = any(f.flaw_id == label and f.confidence >= floor for f in flags)
        per_class[label]["tp" if hit else "fn"] += 1
        if not hit:
            misses.append(entry["file"])
        for f in flags:
            if f.flaw_id in per_class and f.flaw_id != label and f.confidence >= floor:
                per_class[f.flaw_id]["fp"] += 1
        # true-flag survival: correct-label flags that reached the verifier
        for f in fused:
            if f.flaw_id == label:
                survival["total"] += 1
                survival["supported"] += id(f) in kept

    clean_entries = [e for e in manifest["clean"] if e["split"] == split]
    n_real = 0
    real_manifest = root / "real" / "MANIFEST.json"
    if split == "heldout" and real_manifest.exists():
        real_clean = json.loads(real_manifest.read_text())["clean"]
        n_real = len(real_clean)
        clean_entries = clean_entries + real_clean
    for entry in clean_entries:
        flags, _, _ = narrative_flags(entry["file"])
        confident = [f for f in flags if f.confidence >= floor]
        for f in confident:
            if f.flaw_id in per_class:
                per_class[f.flaw_id]["fp"] += 1
        if confident:
            clean_fp_files.append(entry["file"])

    g3 = {}
    neg = json.loads((root / "negative" / "MANIFEST.json").read_text())
    for f in neg["flags"]:
        if f["split"] != split:
            continue
        target = _VerifyTarget(
            flaw_id=f["flaw_id"],
            claim_cell=f["claim_span"]["cell"], claim_quote=f["claim_span"]["quote"],
            code_cell=f["code_span"]["cell"], code_quote=f["code_span"]["quote"],
        )
        verdict = verify_finding(target, parse_notebook(root / f["source_file"]), ver_backend)
        r = g3.setdefault(f["recipe"], {"killed": 0, "total": 0})
        r["total"] += 1
        r["killed"] += not verdict.supported
    for r in g3.values():
        r["kill_rate"] = r["killed"] / r["total"]

    return {
        "date": date.today().isoformat(),
        "corpus_built": manifest["built"],
        "split": split,
        "confidence_floor": floor,
        "detector": {"provider": det_backend.provider, "model": det_backend.model,
                     "kind": det_backend.kind},
        "verifier": {"provider": ver_backend.provider, "model": ver_backend.model,
                     "kind": ver_backend.kind},
        # computed after the run: a replay backend that served any call from
        # disk has flipped ineligible by now; models must also be the pinned
        # ones (per the spec) or a swapped-model run could pass silently
        "gate_evidence": bool(
            det_backend.gate_eligible and ver_backend.gate_eligible
            and det_backend.model == PINNED_DETECTOR_MODEL
            and ver_backend.model == PINNED_VERIFIER_MODEL
        ),
        "usage": {"detector": getattr(det_backend, "usage", None),
                  "verifier": getattr(ver_backend, "usage", None)},
        "n_clean": len(clean_entries),
        "n_clean_real": n_real,
        "narrative_classes": {c: _prf1(v) for c, v in per_class.items()},
        "clean_fp_rate": len(clean_fp_files) / len(clean_entries) if clean_entries else None,
        "clean_fp_files": clean_fp_files,
        "clean_fp_caveat": REAL_FP_CAVEAT if n_real else None,
        "dropped_ungrounded": {
            "dropped": n_dropped,
            "raw_findings": n_raw_findings,
            "rate": n_dropped / n_raw_findings if n_raw_findings else None,
        },
        "g3_per_recipe": g3,
        "true_flag_survival": {
            **survival,
            "rate": survival["supported"] / survival["total"] if survival["total"] else None,
        },
        "missed_mutants": misses,
    }


def render_llm_report(results: dict) -> str:
    def fmt(x):
        return f"{x:.2f}" if x is not None else "—"

    det, ver = results["detector"], results["verifier"]
    lines = [
        f"# Wald LLM eval — {results['date']} "
        f"(corpus built {results['corpus_built']}, split {results['split']})",
        "",
        f"Detector {det['provider']}/{det['model']} ({det['kind']}), "
        f"verifier {ver['provider']}/{ver['model']} ({ver['kind']}). "
        f"Confidence floor {results['confidence_floor']}. "
        f"Gate evidence: {results['gate_evidence']}.",
        "",
        "## Narrative classes (layer B, full pipeline)",
        "",
        "| class | TP | FN | FP | precision | recall | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c, r in results["narrative_classes"].items():
        lines.append(
            f"| {c} | {r['tp']} | {r['fn']} | {r['fp']} | "
            f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} |"
        )
    d = results["dropped_ungrounded"]
    fp_rate = results["clean_fp_rate"]
    lines += [
        "",
        f"Clean FP rate ({results['n_clean']} notebooks, "
        f"{results['n_clean_real']} real): "
        + (f"{fp_rate:.1%} ({len(results['clean_fp_files'])} files)"
           if fp_rate is not None else "—"),
        f"Dropped ungrounded: {d['dropped']}/{d['raw_findings']} raw findings"
        + (f" ({d['rate']:.1%})" if d["rate"] is not None else ""),
        "",
        "## G3 — seeded false flags (verifier kill rate per recipe)",
        "",
    ]
    for recipe, r in sorted(results["g3_per_recipe"].items()):
        lines.append(f"- {recipe}: killed {r['killed']}/{r['total']} ({r['kill_rate']:.0%})")
    s = results["true_flag_survival"]
    lines.append(
        f"\nTrue-flag survival on the same run: {s['supported']}/{s['total']}"
        + (f" ({s['rate']:.0%})" if s["rate"] is not None else "")
    )
    if results["missed_mutants"]:
        lines += ["", "## Missed mutants"]
        lines += [f"- {m}" for m in results["missed_mutants"]]
    usage = results["usage"]
    lines += [
        "",
        "## Honest caveats",
        "- Narrative mutants are template prose from one author; the phrasing "
        "holdout detects memorization but cannot rule it out.",
    ]
    if results["clean_fp_caveat"]:
        lines.append(f"- {results['clean_fp_caveat']}.")
    lines.append(
        f"- Token/cost totals: detector {usage['detector']}, verifier "
        f"{usage['verifier']} (None = backend exposes no usage counters)."
    )
    return "\n".join(lines)


def run_llm_eval(corpus_root: str | Path, det_backend, ver_backend,
                 split: str = "dev", out_dir: str | Path = "evals") -> dict:
    results = evaluate_narrative(corpus_root, det_backend, ver_backend, split=split)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{results['date']}-llm-eval.json").write_text(json.dumps(results, indent=2))
    (out / f"{results['date']}-llm-eval.md").write_text(render_llm_report(results))
    return results
