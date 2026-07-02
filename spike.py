"""De-risk spike (go/no-go): dataflow detection + mutation verify, end to end."""

from wald import execute as ex
from wald.corpus import churn_notebook
from wald.detect import run_static
from wald.ingest import from_nbnode
from wald.mutate import FitBeforeSplitMutation

print("== 1. build + execute clean churn notebook ==")
clean = ex.execute(churn_notebook(11))

print("== 2. static detectors on CLEAN (expect: no flags) ==")
clean_flags = run_static(from_nbnode(clean))
for f in clean_flags:
    print(f"  FLAG {f.flaw_id} conf={f.confidence} cell={f.cell}:{f.line} :: {f.evidence}")
print(f"  -> {len(clean_flags)} flags")

print("== 3. mutate (fit before split) + detect (expect: leakage flag) ==")
mutation = FitBeforeSplitMutation()
assert mutation.applicable(clean)
mutant = mutation.apply(clean, seed=0)
print("  mutated split cell:")
for line in mutant.cells[5]["source"].splitlines():
    print("   |", line)
mutant_flags = run_static(from_nbnode(mutant))
for f in mutant_flags:
    print(f"  FLAG {f.flaw_id} conf={f.confidence} cell={f.cell}:{f.line} :: {f.evidence}")

print("== 4. mechanical verify: mutant (expect seen==total) vs clean control ==")
ok, evidence = mutation.verify(mutant)
print(f"  mutant verify: {ok} {evidence}")
probe = "print('WALD_VERIFY_LEAK', int(getattr(scaler, 'n_samples_seen_', -1)), int(len(y)))"
clean_exec = ex.execute(ex.with_appended_code_cell(clean, probe))
line = ex.stdout_lines(clean_exec, "WALD_VERIFY_LEAK")[-1]
_, seen, total = line.split()
print(f"  clean control: scaler saw {seen}/{total} rows (must be <)")

assert len([f for f in clean_flags if f.confidence >= 0.8]) == 0, "FP on clean!"
assert any(f.flaw_id == "leakage-fit-before-split" for f in mutant_flags), "leakage missed!"
assert ok, "mutation verify failed!"
assert int(seen) < int(total), "clean control invalid!"
print("\nSPIKE PASSED: dataflow detection + CST mutation + execution verify all work.")
