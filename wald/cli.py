"""wald CLI: check / eval / corpus."""

from __future__ import annotations

import argparse
import sys

from .detect import DEFAULT_CONFIDENCE_FLOOR, run_static
from .ingest import parse_notebook
from .report import exit_code, to_json, to_markdown


def cmd_check(args) -> int:
    worst = 0
    for path in args.notebooks:
        flags = run_static(parse_notebook(path))
        if args.format == "json":
            print(to_json(path, flags, args.floor))
        else:
            print(to_markdown(path, flags, args.floor))
        worst = max(worst, exit_code(flags, args.floor, args.severity_gate))
    return worst


def cmd_eval(args) -> int:
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
    p_check.set_defaults(func=cmd_check)

    p_eval = sub.add_parser("eval", help="run detectors over the corpus, write dated report")
    p_eval.add_argument("--corpus", default="corpus")
    p_eval.add_argument("--out", default="evals")
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
