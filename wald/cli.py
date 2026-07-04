"""wald CLI: check / eval / corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detect import DEFAULT_CONFIDENCE_FLOOR, run_static
from .ingest import parse_notebook
from .report import exit_code, to_json, to_markdown


def _llm_backends(replay_dir):
    from .llm import AnthropicBackend, OpenAIBackend, ReplayBackend

    det, ver = AnthropicBackend(), OpenAIBackend()
    if replay_dir:
        det = ReplayBackend(Path(replay_dir) / "detector", det)
        ver = ReplayBackend(Path(replay_dir) / "verifier", ver)
    return det, ver


def _heldout_refusal(path, det, ver) -> str | None:
    """Held-out corpus notebooks are gate-only (m2 item 11): block `check
    --llm` on them so the eval guard can't be sidestepped notebook-by-notebook
    with a replay/agent backend."""
    if det.kind == "api" and ver.kind == "api":
        return None
    p = Path(path).resolve()
    for anc in (p.parent, *p.parent.parents):
        manifest_path = anc / "MANIFEST.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        rel = str(p.relative_to(anc))
        entries = manifest.get("clean", []) + manifest.get("mutants", [])
        if any(e.get("file") == rel and e.get("split") == "heldout" for e in entries):
            return f"{path}: held-out corpus notebook is gate-only, refusing --llm check"
        break
    return None


def cmd_check(args) -> int:
    if args.llm:
        from .fuse import run_full

        det, ver = _llm_backends(args.replay_dir)
    worst = 0
    for path in args.notebooks:
        if args.llm:
            refusal = _heldout_refusal(path, det, ver)
            if refusal:
                print(refusal, file=sys.stderr)
                return 2
        nb = parse_notebook(path)
        flags = run_full(nb, det, ver) if args.llm else run_static(nb)
        if args.format == "json":
            print(to_json(path, flags, args.floor))
        else:
            print(to_markdown(path, flags, args.floor))
        worst = max(worst, exit_code(flags, args.floor, args.severity_gate))
    return worst


def cmd_eval(args) -> int:
    if args.llm:
        from .eval import run_llm_eval

        det, ver = _llm_backends(args.replay_dir)
        results = run_llm_eval(args.corpus, det, ver, split=args.split, out_dir=args.out)
        print(f"llm eval written to {args.out}/{results['date']}-llm-eval.md")
        for c, r in results["narrative_classes"].items():
            f1 = f"{r['f1']:.2f}" if r["f1"] is not None else "—"
            print(f"  {c}: F1 {f1}")
        fp = results["clean_fp_rate"]
        print(f"  clean FP rate: {fp:.1%}" if fp is not None else "  clean FP rate: —")
        for recipe, r in sorted(results["g3_per_recipe"].items()):
            print(f"  G3 {recipe}: kill {r['killed']}/{r['total']}")
        print(f"  gate evidence: {results['gate_evidence']}")
        return 0

    from .eval import run_eval

    results = run_eval(args.corpus, args.out)
    print(f"eval written to {args.out}/{results['date']}-eval.md")
    for c, r in results["static_classes"].items():
        p = f"{r['precision']:.2f}" if r["precision"] is not None else "—"
        rc = f"{r['recall']:.2f}" if r["recall"] is not None else "—"
        print(f"  {c}: precision {p} recall {rc}")
    print(f"  clean FP rate: {results['clean_fp_rate']:.1%}")
    return 0


def cmd_corpus_build(args) -> int:
    from .corpus import build_corpus

    build_corpus(args.root, seeds=tuple(args.seeds))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wald", description="Statistical-integrity linter for notebooks.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="lint notebook(s); exit 2 on high-severity findings")
    p_check.add_argument("notebooks", nargs="+")
    p_check.add_argument("--format", choices=["md", "json"], default="md")
    p_check.add_argument("--floor", type=float, default=DEFAULT_CONFIDENCE_FLOOR)
    p_check.add_argument("--severity-gate", choices=["medium", "high"], default="high")
    p_check.add_argument("--llm", action="store_true",
                         help="add the narrative layer (needs API keys)")
    p_check.add_argument("--replay-dir", help="record/replay LLM responses here")
    p_check.set_defaults(func=cmd_check)

    p_eval = sub.add_parser("eval", help="run detectors over the corpus, write dated report")
    p_eval.add_argument("--corpus", default="corpus")
    p_eval.add_argument("--out", default="evals")
    p_eval.add_argument("--llm", action="store_true",
                        help="narrative-layer eval (needs API keys)")
    p_eval.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p_eval.add_argument("--replay-dir", help="record/replay LLM responses here")
    p_eval.set_defaults(func=cmd_eval)

    p_corpus = sub.add_parser("corpus", help="corpus operations")
    corpus_sub = p_corpus.add_subparsers(dest="corpus_command", required=True)
    p_build = corpus_sub.add_parser("build", help="build clean notebooks + verified mutants")
    p_build.add_argument("--root", default="corpus")
    p_build.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13, 14])
    p_build.set_defaults(func=cmd_corpus_build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
