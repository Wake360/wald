"""wald CLI: check / eval / corpus."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from nbformat.reader import NotJSONError

from .dataflow import analyze
from .detect import DEFAULT_CONFIDENCE_FLOOR, run_static
from .fuse import run_full_traced
from .ingest import parse_notebook
from .report import _colorize, exit_code, parse_warning, report_obj, to_markdown, to_sarif

# environment variable each api backend needs before it can make a request
_KEY_BY_PROVIDER = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


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
    with a replay/agent backend. Real-world corpus notebooks (corpus/real/*)
    are held-out material too — eval folds them into the heldout clean set and
    their manifest carries no per-entry split field, so they are refused by
    living under a `real/` manifest."""
    if det.kind == "api" and ver.kind == "api":
        return None
    p = Path(path).resolve()
    for anc in (p.parent, *p.parent.parents):
        manifest_path = anc / "MANIFEST.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        entries = manifest.get("clean", []) + manifest.get("mutants", [])
        for e in entries:
            f = e.get("file")
            # manifest paths are relative to the manifest dir (corpus/clean/..)
            # or to the corpus root (real/MANIFEST.json prefixes with "real/")
            if f is None or p not in ((anc / f).resolve(), (anc.parent / f).resolve()):
                continue
            if e.get("split") == "heldout" or anc.name == "real":
                return f"{path}: held-out corpus notebook is gate-only, refusing --llm check"
        break
    return None


def _missing_llm_keys(*backends) -> list[str]:
    """Env vars an api backend needs but that are unset (replay/agent need none)."""
    missing = []
    for b in backends:
        if b.kind == "api":
            env = _KEY_BY_PROVIDER.get(b.provider)
            if env and not os.environ.get(env):
                missing.append(env)
    return sorted(set(missing))


class _EmptyDir(Exception):
    """A directory argument contained no notebooks."""


def _expand_notebooks(paths: list[str]) -> list[str]:
    """Replace directory arguments with their *.ipynb files, recursive and
    sorted, skipping .ipynb_checkpoints copies. File arguments pass through."""
    expanded = []
    for path in paths:
        p = Path(path)
        if not p.is_dir():
            expanded.append(path)
            continue
        found = sorted(
            str(q) for q in p.rglob("*.ipynb") if ".ipynb_checkpoints" not in q.parts
        )
        if not found:
            raise _EmptyDir(path)
        expanded.extend(found)
    return expanded


def _input_error(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "no such file"
    if isinstance(exc, IsADirectoryError):
        return "is a directory, not a notebook"
    if isinstance(exc, UnicodeDecodeError):
        return "not valid UTF-8 text"
    if isinstance(exc, NotJSONError):
        return "not a valid notebook (invalid JSON)"
    return str(exc)


def cmd_check(args) -> int:
    if not 0.0 <= args.floor <= 1.0:
        print(f"wald: --floor must be between 0 and 1 (got {args.floor})", file=sys.stderr)
        return 3
    det = ver = None
    if args.llm:
        det, ver = _llm_backends(args.replay_dir)
        missing = _missing_llm_keys(det, ver)
        if missing:
            print(f"wald: --llm needs {' and '.join(missing)} set in the environment",
                  file=sys.stderr)
            return 3
    try:
        notebooks = _expand_notebooks(args.notebooks)
    except _EmptyDir as exc:
        print(f"wald: {exc}: no .ipynb files found", file=sys.stderr)
        return 3
    # interactive-only chrome; the piped md/json/sarif bytes stay untouched
    color = args.format == "md" and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    progress = args.llm and sys.stderr.isatty()

    def close_progress():
        # end the \r-overwritten line so errors/summaries start on a fresh one
        if progress:
            print(file=sys.stderr)

    reports = []
    sarif_entries = []
    worst = 0
    n_high = n_med = n_clean = 0
    for i, path in enumerate(notebooks, 1):
        if progress:
            print(f"\rchecking {i}/{len(notebooks)} {path}", end="", file=sys.stderr,
                  flush=True)
        if args.llm:
            refusal = _heldout_refusal(path, det, ver)
            if refusal:
                close_progress()
                print(f"wald: {refusal}", file=sys.stderr)
                return 3
        try:
            nb = parse_notebook(path)
            flow = analyze(nb)
            if args.llm:
                flags, narrative, _ = run_full_traced(nb, det, ver)
                # a detector/verifier outage fails closed inside detect_narrative
                # (empty result tagged in `dropped`); surface it so a broken
                # backend can never read like a clean notebook.
                backend_error = next(
                    (d for d in narrative.dropped if d.startswith("backend error:")), None
                )
                if backend_error is not None:
                    close_progress()
                    print(f"wald: narrative layer failed: {backend_error}", file=sys.stderr)
                    return 3
            else:
                flags = run_static(nb, flow)
        except Exception as exc:
            close_progress()
            print(f"wald: {path}: {_input_error(exc)}", file=sys.stderr)
            return 3
        warning = parse_warning(len(flow.parse_errors), len(nb.code_cells))
        if args.format == "json":
            reports.append(report_obj(path, flags, args.floor, args.severity_gate, warning))
        elif args.format == "sarif":
            sarif_entries.append((path, flags))
        else:
            text = to_markdown(path, flags, args.floor, warning, args.llm)
            print(_colorize(text) if color else text)
        confident = [f for f in flags if f.confidence >= args.floor]
        if any(f.severity == "high" for f in confident):
            n_high += 1
        elif any(f.severity == "medium" for f in confident):
            n_med += 1
        else:
            n_clean += 1
        worst = max(worst, exit_code(flags, args.floor, args.severity_gate))
    close_progress()
    if args.format == "json":
        # one bare object for a single notebook (back-compat), an array for many
        print(json.dumps(reports[0] if len(reports) == 1 else reports, indent=2))
    elif args.format == "sarif":
        print(to_sarif(sarif_entries, args.floor))
    elif len(notebooks) > 1 and sys.stdout.isatty():
        print(f"checked {len(notebooks)} notebooks: {n_high} high, {n_med} medium, "
              f"{n_clean} clean", file=sys.stderr)
    return worst


def cmd_eval(args) -> int:
    if args.llm:
        from .eval import run_llm_eval

        det, ver = _llm_backends(args.replay_dir)
        missing = _missing_llm_keys(det, ver)
        if missing:
            print(f"wald: --llm needs {' and '.join(missing)} set in the environment",
                  file=sys.stderr)
            return 3
        results = run_llm_eval(args.corpus, det, ver, split=args.split, out_dir=args.out)
        print(f"llm eval written to {args.out}/{results['date']}-llm-eval-{results['split']}.md")
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
    try:
        _version = importlib.metadata.version("wald-lint")
    except importlib.metadata.PackageNotFoundError:
        _version = "unknown"
    parser.add_argument("-V", "--version", action="version", version=f"wald {_version}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser(
        "check",
        help="lint notebook(s); exit 0 clean / 1 medium / 2 high-severity / 3 input or usage error",
        epilog="exit codes: 0 clean, 1 medium, 2 high, 3 input or usage error",
    )
    p_check.add_argument("notebooks", nargs="+",
                         help="notebook files or directories (searched recursively "
                              "for *.ipynb, skipping .ipynb_checkpoints)")
    p_check.add_argument("--format", choices=["md", "json", "sarif"], default="md",
                         help="json emits one object for a single notebook, a JSON array for "
                              "several; sarif emits one SARIF 2.1.0 log for the whole invocation")
    p_check.add_argument("--floor", type=float, default=DEFAULT_CONFIDENCE_FLOOR,
                         help="confidence floor in [0, 1] (default: %(default)s); findings below "
                              "it move to Candidates instead of Flags")
    p_check.add_argument("--severity-gate", choices=["medium", "high"], default="high",
                         help="exit 2 at or above this severity (default: %(default)s); "
                              "confident findings below the gate exit 1")
    p_check.add_argument("--llm", action="store_true",
                         help="add the narrative layer (needs API keys)")
    p_check.add_argument("--replay-dir", help="record/replay LLM responses here")
    p_check.set_defaults(func=cmd_check)

    p_eval = sub.add_parser("eval", help="run detectors over the corpus, write dated report")
    p_eval.add_argument("--corpus", default="corpus",
                        help="corpus root directory (default: %(default)s)")
    p_eval.add_argument("--out", default="evals",
                        help="directory for dated eval reports (default: %(default)s)")
    p_eval.add_argument("--llm", action="store_true",
                        help="narrative-layer eval (needs API keys)")
    p_eval.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p_eval.add_argument("--replay-dir", help="record/replay LLM responses here")
    p_eval.set_defaults(func=cmd_eval)

    p_corpus = sub.add_parser("corpus", help="corpus operations")
    corpus_sub = p_corpus.add_subparsers(dest="corpus_command", required=True)
    p_build = corpus_sub.add_parser("build", help="build clean notebooks + verified mutants")
    p_build.add_argument("--root", default="corpus",
                         help="corpus root directory (default: %(default)s)")
    p_build.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13, 14],
                         help="dev-split base seeds, one clean notebook per family "
                              "per seed (default: %(default)s)")
    p_build.set_defaults(func=cmd_corpus_build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
