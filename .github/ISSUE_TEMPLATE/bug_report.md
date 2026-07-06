---
name: Bug report
about: Wald flagged wrong, missed a flaw, or crashed
title: ""
labels: bug
---

**Notebook**

Attach the notebook that reproduces the issue, or a minimal `.ipynb` that
still reproduces it (strip unrelated cells and large outputs).

**Wald version**

Output of `wald --version` (or `pip show wald-lint`).

**Static-only or `--llm`**

Which mode you ran. If `--llm`, include which provider(s).

**Full Wald output**

The complete output of the `wald check` invocation, not a trimmed excerpt.
Use `--format json` if that's easier to paste in full.
