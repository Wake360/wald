---
name: Feature request
about: Propose a new flaw class or other feature
title: ""
labels: enhancement
---

**Definition**

What the flaw is, precisely enough to write a detector against — not just
a name. What code, stored output, or prose pattern signals it.

**Book anchor**

The statistics/ML reference that names this error (textbook, paper,
section). Wald's taxonomy cites one per class; see `wald/taxonomy/flaws.yaml`
for the existing format.

**Mutation recipe sketch**

How a clean notebook would be mutated to inject this flaw, and how the
mutation would be mechanically verified (not just pattern-matched) —
see `CONTRIBUTING.md` for the shape a flaw-class contribution needs.
