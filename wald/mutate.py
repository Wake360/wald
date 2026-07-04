"""Mutation engine: inject named flaws into clean notebooks, with proof.

Every mutation must mechanically verify that the flaw is really present in
the mutant (execution fingerprint or AST fact). A mutant without proof
never enters the corpus — otherwise the whole eval stands on sand.

One mutation per notebook (isolation of measurement). Cell indices in the
notebook's wald metadata refer to the clean base; mutations either edit
in place or insert cells and never rely on post-insertion indices.
"""

from __future__ import annotations

import copy
import random
import re

import libcst as cst
import nbformat

from . import execute as ex
from .dataflow import analyze
from .detect import CORRECTION_RE, LOOP_TEST_WEIGHT, MULTIPLE_TESTING_THRESHOLD, TEST_FUNCS
from .ingest import from_nbnode


def meta(nb_node) -> dict:
    return dict(nb_node.metadata.get("wald", {}))


def _cell_stmts(source: str) -> tuple[cst.Module, list]:
    module = cst.parse_module(source)
    return module, list(module.body)


def _stmt_code(module: cst.Module, stmt) -> str:
    return module.code_for_node(stmt)


def _replace_cell(nb_node, index: int, new_source: str):
    nb = copy.deepcopy(nb_node)
    nb.cells[index]["source"] = new_source
    nb.cells[index]["outputs"] = []
    nb.cells[index]["execution_count"] = None
    return nb


def _insert_code_cell(nb_node, index: int, source: str):
    nb = copy.deepcopy(nb_node)
    nb.cells.insert(index, nbformat.v4.new_code_cell(source))
    return nb


def _replace_markdown(nb_node, index: int, text: str):
    nb = copy.deepcopy(nb_node)
    nb.cells[index]["source"] = text
    return nb


# phrasing variants 0-1 are dev-only; 2-4 exist only in held-out mutants,
# and variant 4 is a rhetorical shape (question-then-answer) never seen in
# dev — memorizing dev phrasings cannot cover it (risk R3)
DEV_VARIANTS = (0, 1)
HELDOUT_VARIANTS = (2, 3, 4)


def phrasing_variant(seed: int, base_seed: int) -> int:
    pool = DEV_VARIANTS if seed < 2 else HELDOUT_VARIANTS
    return pool[(seed + base_seed) % len(pool)]


class Mutation:
    flaw_id: str = ""
    POOL: tuple[str, ...] = ()

    def applicable(self, nb_node) -> bool:
        raise NotImplementedError

    def apply(self, nb_node, seed: int):
        raise NotImplementedError

    def verify(self, mutated_node) -> tuple[bool, dict]:
        raise NotImplementedError

    def conclusion(self, nb_node, seed: int) -> str | None:
        if not self.POOL:
            return None
        return self.POOL[phrasing_variant(seed, meta(nb_node)["seed"])]


class FitBeforeSplitMutation(Mutation):
    """Move the scaler fit before train_test_split, consuming the full frame."""

    flaw_id = "leakage-fit-before-split"

    def applicable(self, nb_node) -> bool:
        m = meta(nb_node)
        if "split_cell" not in m:
            return False
        src = nb_node.cells[m["split_cell"]]["source"]
        return "train_test_split" in src and "fit_transform" in src

    def apply(self, nb_node, seed: int):
        m = meta(nb_node)
        idx = m["split_cell"]
        module, stmts = _cell_stmts(nb_node.cells[idx]["source"])

        split_stmt = fit_stmt = transform_stmt = scaler_def = None
        for s in stmts:
            code = _stmt_code(module, s)
            if "train_test_split" in code:
                split_stmt = s
            elif ".fit_transform(" in code:
                fit_stmt = s
            elif ".transform(" in code:
                transform_stmt = s
            elif "StandardScaler()" in code:
                scaler_def = s
        if not all([split_stmt, fit_stmt, transform_stmt, scaler_def]):
            raise ValueError("canonical split cell not found")

        split_code = _stmt_code(module, split_stmt)
        call_args = re.search(r"train_test_split\(\s*(\w+)\s*,", split_code)
        pre_name = call_args.group(1)
        new_fit = cst.parse_statement(f"{pre_name} = scaler.fit_transform({pre_name})\n")

        rest = [s for s in stmts if s not in {split_stmt, fit_stmt, transform_stmt, scaler_def}]
        new_body = [scaler_def, new_fit, split_stmt] + rest
        new_source = module.with_changes(body=new_body).code.rstrip("\n")
        return _replace_cell(nb_node, idx, new_source)

    def verify(self, mutated_node) -> tuple[bool, dict]:
        probe = (
            "print('WALD_VERIFY_LEAK', int(getattr(scaler, 'n_samples_seen_', -1)), int(len(y)))"
        )
        executed = ex.execute(ex.with_appended_code_cell(mutated_node, probe))
        lines = ex.stdout_lines(executed, "WALD_VERIFY_LEAK")
        if not lines:
            return False, {"reason": "no verify output"}
        _, seen, total = lines[-1].split()
        ok = int(seen) == int(total) and int(seen) > 0
        return ok, {"scaler_samples_seen": int(seen), "total_rows": int(total)}


class MultipleTestingMutation(Mutation):
    """Insert an uncorrected t-test screen over many columns, plus a
    conclusion that cites the 'significant' hits."""

    flaw_id = "testing-multiple-uncorrected"

    def applicable(self, nb_node) -> bool:
        m = meta(nb_node)
        return bool(m.get("num_cols")) and (m.get("binary_col") or m.get("target_col"))

    def apply(self, nb_node, seed: int):
        m = meta(nb_node)
        rng = random.Random(seed)
        cols = list(m["num_cols"])
        k = min(len(cols), 8)
        tested = rng.sample(cols, k)

        if m.get("binary_col"):
            col, v1, v0 = m["binary_col"], m["binary_values"][0], m["binary_values"][1]
            a = f"df[df[{col!r}] == {v1!r}][c]"
            b = f"df[df[{col!r}] == {v0!r}][c]"
        else:
            t = m["target_col"]
            a = f"df[df[{t!r}] > df[{t!r}].median()][c]"
            b = f"df[df[{t!r}] <= df[{t!r}].median()][c]"

        code = (
            "from scipy.stats import ttest_ind\n"
            f"cols_to_test = {tested!r}\n"
            "significant = []\n"
            "for c in cols_to_test:\n"
            f"    stat, p = ttest_ind({a}, {b})\n"
            "    if p < 0.05:\n"
            "        significant.append(c)\n"
            "print('tested', len(cols_to_test), 'columns; significant:', significant)"
        )
        md = (
            "Screening all available metrics, we identified the significant "
            "drivers listed above — these differ reliably between groups and "
            "should be prioritized."
        )
        idx = m["conclusion_cell"]
        nb = _insert_code_cell(nb_node, idx, code)
        nb.cells.insert(idx + 1, nbformat.v4.new_markdown_cell(md))
        return nb

    def verify(self, mutated_node) -> tuple[bool, dict]:
        nb = from_nbnode(mutated_node)
        flow = analyze(nb)
        sites = [c for c in flow.calls if c.name in TEST_FUNCS]
        effective_n = sum(LOOP_TEST_WEIGHT if c.loop_depth > 0 else 1 for c in sites)
        corrected = bool(CORRECTION_RE.search(nb.full_source()))
        ok = effective_n > MULTIPLE_TESTING_THRESHOLD and not corrected
        if ok:  # must also actually run
            ex.execute(mutated_node)
        return ok, {"effective_tests": effective_n, "correction_present": corrected}


class _DropImportNames(cst.CSTTransformer):
    def __init__(self, names: set[str]):
        self.names = names

    def leave_ImportFrom(self, original, updated):
        if isinstance(updated.names, cst.ImportStar):
            return updated
        kept = [a for a in updated.names if a.name.value not in self.names]
        if not kept:
            return cst.RemoveFromParent()
        kept[-1] = kept[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
        return updated.with_changes(names=kept)


class BaserateAccuracyMutation(Mutation):
    """Strip AUC, keep accuracy as the only metric, rewrite the conclusion
    to an accuracy brag with no imbalance mention."""

    flaw_id = "baserate-accuracy-imbalanced"

    def applicable(self, nb_node) -> bool:
        m = meta(nb_node)
        return bool(m.get("imbalanced")) and "metrics_cell" in m and "imports_cell" in m

    def apply(self, nb_node, seed: int):
        m = meta(nb_node)
        # 1) metrics cell: drop auc/proba lines, accuracy-only print
        idx = m["metrics_cell"]
        module, stmts = _cell_stmts(nb_node.cells[idx]["source"])
        new_body = []
        for s in stmts:
            code = _stmt_code(module, s)
            if "roc_auc" in code or "predict_proba" in code:
                continue
            if code.strip().startswith("print("):
                s = cst.parse_statement('print(f"accuracy={acc:.3f}")\n')
            new_body.append(s)
        nb = _replace_cell(nb_node, idx, module.with_changes(body=new_body).code.rstrip("\n"))

        # 2) imports cell: remove roc_auc_score so no metric token survives
        imp_idx = m["imports_cell"]
        imp_module = cst.parse_module(nb.cells[imp_idx]["source"])
        new_imports = imp_module.visit(_DropImportNames({"roc_auc_score"}))
        nb = _replace_cell(nb, imp_idx, new_imports.code.rstrip("\n"))

        # 3) conclusion: accuracy overclaim, imbalance mention gone
        md = (
            "The classifier is highly accurate on held-out data — "
            "performance is strong and the model is ready to use."
        )
        return _replace_markdown(nb, m["conclusion_cell"], md)

    def verify(self, mutated_node) -> tuple[bool, dict]:
        src = "\n".join(
            c["source"] for c in mutated_node.cells if c["cell_type"] == "code"
        )
        if "roc_auc" in src:
            return False, {"reason": "auc still present"}
        probe = (
            "import pandas as _pd\n"
            "_maj = float(_pd.Series(y_te).value_counts(normalize=True).max())\n"
            "print('WALD_VERIFY_BASERATE', _maj, float(acc))"
        )
        executed = ex.execute(ex.with_appended_code_cell(mutated_node, probe))
        lines = ex.stdout_lines(executed, "WALD_VERIFY_BASERATE")
        if not lines:
            return False, {"reason": "no verify output"}
        _, maj, acc = lines[-1].split()
        ok = float(maj) >= float(acc) - 0.02
        return ok, {"majority_baseline": float(maj), "reported_accuracy": float(acc)}


POPULATION_RE = re.compile(
    r"\ball\b|\bentire\b|\bwhole\b|\bevery\b|population|customer value|customer base",
    re.IGNORECASE,
)
SCOPING_RE = re.compile(r"active|retained|surviv|complet", re.IGNORECASE)


class SurvivorshipMutation(Mutation):
    """Filter the cohort to survivors, then state the conclusion about the
    whole population."""

    flaw_id = "selection-survivorship-cohort"

    FILTERS = (
        'df = df[df["status"] == "active"]',
        "df = df.query(\"status == 'active'\")",
    )
    POOL = (
        "Average customer LTV grows steadily across signup cohorts — the "
        "product increases customer value over time.",
        "LTV per cohort keeps climbing quarter over quarter: every new "
        "cohort of customers is worth more than the last.",
        "The upward trend in cohort lifetime value means our customer base "
        "as a whole is becoming more valuable.",
        "Each successive signup quarter delivers higher average LTV, so "
        "the business now extracts more value from all its customers.",
        "Is customer value improving? The cohort table answers clearly: "
        "across the entire customer population, later signups are worth "
        "more.",
    )

    def applicable(self, nb_node) -> bool:
        m = meta(nb_node)
        return m.get("status_col") == "status" and "agg_cell" in m and "conclusion_cell" in m

    def apply(self, nb_node, seed: int):
        m = meta(nb_node)
        filter_line = self.FILTERS[seed % 2]
        if seed % 2 == 0:
            idx = m["agg_cell"]
            nb = _replace_cell(nb_node, idx, filter_line + "\n" + nb_node.cells[idx]["source"])
        else:
            nb = _insert_code_cell(nb_node, m["agg_cell"], filter_line)
        # conclusion index unshifted with the prepended filter; +1 after insert
        concl = m["conclusion_cell"] + (0 if seed % 2 == 0 else 1)
        return _replace_markdown(nb, concl, self.conclusion(nb_node, seed))

    def verify(self, mutated_node) -> tuple[bool, dict]:
        src = "\n".join(
            c["source"] for c in mutated_node.cells if c["cell_type"] == "code"
        )
        if not any(f in src for f in self.FILTERS):
            return False, {"reason": "filter missing"}
        md = "\n".join(
            c["source"] for c in mutated_node.cells if c["cell_type"] == "markdown"
        )
        population_claim = bool(POPULATION_RE.search(md))
        scoped = bool(SCOPING_RE.search(md))
        ok = population_claim and not scoped
        if ok:
            ex.execute(mutated_node)
        return ok, {"population_claim": population_claim, "scoped_language": scoped}


EFFECT_SIZE_RE = re.compile(
    r"effect[ -]size|cohen|confidence interval|\bci\b|\bd\s*=", re.IGNORECASE
)
RECOMMEND_RE = re.compile(
    r"\b(roll|ship|launch|recommend|adopt|deploy)", re.IGNORECASE
)


class SignificanceMeaninglessMutation(Mutation):
    """Blow the sample up so a ~0.04 SD effect is p<0.05 near-certain; the
    Cohen's d computation stays in code — the flaw is the prose citing only
    p and recommending rollout."""

    flaw_id = "significance-meaningless"

    POOL = (
        "The checkout redesign produced a statistically significant "
        "improvement in session time (p < 0.05). We recommend rolling "
        "variant B out to all users.",
        "Variant B beat A on the primary metric with p < 0.05. The result "
        "is statistically significant, so B should be shipped to 100% of "
        "traffic.",
        "With the p-value comfortably below the 0.05 threshold, the "
        "experiment is decisive: launch the new checkout for the entire "
        "user base.",
        "Our significance test cleared the bar (p < 0.05), which settles "
        "it — the data supports adopting variant B as the default "
        "experience going forward.",
        "Did the redesign work? The statistics say yes: p < 0.05 on the "
        "primary metric. Roll variant B out to everyone.",
    )

    def applicable(self, nb_node) -> bool:
        m = meta(nb_node)
        if "datagen_cell" not in m or "conclusion_cell" not in m:
            return False
        src = nb_node.cells[m["datagen_cell"]]["source"]
        return "session_minutes" in src and "n = 2400" in src

    def apply(self, nb_node, seed: int):
        m = meta(nb_node)
        idx = m["datagen_cell"]
        # 20k/group and a 0.17-minute lift: ~0.045 SD (gamma(3, 2.2) has
        # sd 3.81), z ~ 4.5, so p<0.05 holds for essentially every seed
        src = nb_node.cells[idx]["source"]
        src = src.replace("n = 2400", "n = 40000")
        src = src.replace('(variant == "B") * 0.5', '(variant == "B") * 0.17')
        nb = _replace_cell(nb_node, idx, src)
        return _replace_markdown(nb, m["conclusion_cell"], self.conclusion(nb_node, seed))

    def verify(self, mutated_node) -> tuple[bool, dict]:
        probe = "print('WALD_VERIFY_SIG', float(p), float(d))"
        executed = ex.execute(ex.with_appended_code_cell(mutated_node, probe))
        lines = ex.stdout_lines(executed, "WALD_VERIFY_SIG")
        if not lines:
            return False, {"reason": "no verify output"}
        _, p, d = lines[-1].split()
        md = "\n".join(
            c["source"] for c in mutated_node.cells if c["cell_type"] == "markdown"
        )
        effect_mentioned = bool(EFFECT_SIZE_RE.search(md))
        recommends = bool(RECOMMEND_RE.search(md))
        ok = float(p) < 0.05 and abs(float(d)) < 0.2 and not effect_mentioned and recommends
        return ok, {
            "p": float(p), "cohens_d": float(d),
            "effect_size_in_markdown": effect_mentioned, "recommendation": recommends,
        }


CAUSAL_RE = re.compile(r"caused|effect of|drove|produced|worked", re.IGNORECASE)
CONTROL_RE = re.compile(r"control|comparison group|untreated", re.IGNORECASE)
EXTREME_SELECTORS = {"nsmallest", "nlargest", "quantile"}


class RegressionToMeanMutation(Mutation):
    """Select an extreme group at period 1 and read its period-2 move toward
    the mean as the effect of a program, with no control group."""

    flaw_id = "regression-to-mean-claim"

    POOL = (
        "Accounts flagged for unusual spend were enrolled in the "
        "spend-stabilization program, and their spend moved back toward "
        "typical levels the next quarter. The program clearly worked — the "
        "intervention caused the correction.",
        "After we enrolled the most extreme spenders in the stabilization "
        "initiative, their quarterly spend normalized. That shift is the "
        "direct effect of the initiative.",
        "The outreach paid off: accounts singled out for atypical spending "
        "returned to ordinary levels once the program took hold, which "
        "shows the program drove the turnaround.",
        "Enrollment in the outreach scheme produced a marked normalization "
        "in the flagged accounts' spending — proof that the scheme changed "
        "spending behavior.",
        "Did the stabilization program work? The numbers say yes — flagged "
        "accounts' spend moved back to normal in the following quarter, a "
        "change the program caused.",
    )

    def applicable(self, nb_node) -> bool:
        m = meta(nb_node)
        return "period_cols" in m and "conclusion_cell" in m

    def apply(self, nb_node, seed: int):
        m = meta(nb_node)
        p1, p2 = m["period_cols"]
        select = [
            f"flagged = df.nsmallest(150, {p1!r})",
            f"flagged = df.nlargest(150, {p1!r})",
            f"flagged = df[df[{p1!r}] <= df[{p1!r}].quantile(0.05)]",
        ][seed % 3]
        code = (
            select + "\n"
            'print("flagged", len(flagged),\n'
            f'      "q1", round(flagged[{p1!r}].mean(), 2),\n'
            f'      "q2", round(flagged[{p2!r}].mean(), 2))'
        )
        idx = m["conclusion_cell"]
        nb = _insert_code_cell(nb_node, idx, code)
        nb.cells.insert(idx + 1, nbformat.v4.new_markdown_cell(self.conclusion(nb_node, seed)))
        return nb

    def verify(self, mutated_node) -> tuple[bool, dict]:
        nb = from_nbnode(mutated_node)
        flow = analyze(nb)
        selections = [c for c in flow.calls if c.name in EXTREME_SELECTORS]
        if len(selections) != 1 or CONTROL_RE.search(nb.full_source()):
            return False, {"reason": "selection/control structure wrong",
                           "selection_calls": len(selections)}
        p1, p2 = meta(mutated_node)["period_cols"]
        probe = (
            f"print('WALD_VERIFY_RTM', float(flagged[{p1!r}].mean()), "
            f"float(flagged[{p2!r}].mean()), float(df[{p1!r}].mean()), "
            f"float(df[{p2!r}].mean()))"
        )
        executed = ex.execute(ex.with_appended_code_cell(mutated_node, probe))
        lines = ex.stdout_lines(executed, "WALD_VERIFY_RTM")
        if not lines:
            return False, {"reason": "no verify output"}
        _, s1, s2, g1, g2 = lines[-1].split()
        moved = abs(float(s2) - float(g2)) < abs(float(s1) - float(g1))
        md = "\n".join(
            c["source"] for c in mutated_node.cells if c["cell_type"] == "markdown"
        )
        causal = bool(CAUSAL_RE.search(md))
        control_lang = bool(CONTROL_RE.search(md))
        ok = moved and causal and not control_lang
        return ok, {
            "selected_q1_mean": float(s1), "selected_q2_mean": float(s2),
            "grand_q1_mean": float(g1), "grand_q2_mean": float(g2),
            "causal_claim": causal, "control_language": control_lang,
        }


MUTATIONS: list[Mutation] = [
    FitBeforeSplitMutation(),
    MultipleTestingMutation(),
    BaserateAccuracyMutation(),
    SurvivorshipMutation(),
    SignificanceMeaninglessMutation(),
    RegressionToMeanMutation(),
]
