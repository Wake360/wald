"""Name-level def-use analysis over notebook code cells (libcst).

Deliberately not a full abstract interpretation: names are tracked
conservatively (dependencies union across reassignments), which is enough
for the stereotypical pandas/sklearn idioms v1 targets and errs toward
flagging (see README "what Wald doesn't see").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from .ingest import ParsedNotebook


def expr_names(node: cst.CSTNode) -> set[str]:
    """Variable names an expression reads. Attribute chains contribute only
    their base name (df.status -> df); attr parts are not variables."""
    names: set[str] = set()

    def walk(n: cst.CSTNode) -> None:
        if isinstance(n, cst.Name):
            names.add(n.value)
        elif isinstance(n, cst.Attribute):
            walk(n.value)
        elif isinstance(n, cst.Call):
            walk(n.func)
            for arg in n.args:
                walk(arg.value)
        else:
            for child in n.children:
                walk(child)

    walk(node)
    return names


def _dotted(node: cst.BaseExpression) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr.value}" if base else None
    return None


@dataclass
class CallSite:
    func: str  # dotted, e.g. "scaler.fit_transform" or "train_test_split"
    name: str  # final segment, e.g. "fit_transform"
    receiver: str | None  # "scaler" for scaler.fit_transform, None for bare calls
    arg_names: set[str]  # names read by positional/keyword args
    cell: int
    line: int  # 1-based within the cell
    loop_depth: int


@dataclass
class AssignEvent:
    targets: set[str]
    sources: set[str]
    cell: int
    line: int
    call: CallSite | None = None  # set when the assigned value is a call


class _CellVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, cell_index: int):
        self.cell = cell_index
        self.loop_depth = 0
        self.calls: list[CallSite] = []
        self.assigns: list[AssignEvent] = []

    def _line(self, node: cst.CSTNode) -> int:
        return self.get_metadata(PositionProvider, node).start.line

    def visit_For(self, node: cst.For) -> None:
        self.loop_depth += 1

    def leave_For(self, node: cst.For) -> None:
        self.loop_depth -= 1

    def visit_While(self, node: cst.While) -> None:
        self.loop_depth += 1

    def leave_While(self, node: cst.While) -> None:
        self.loop_depth -= 1

    def visit_Call(self, node: cst.Call) -> None:
        dotted = _dotted(node.func)
        if dotted is None:
            return
        receiver = None
        if isinstance(node.func, cst.Attribute):
            receiver = _dotted(node.func.value)
        args: set[str] = set()
        for arg in node.args:
            args |= expr_names(arg.value)
        self.calls.append(
            CallSite(
                func=dotted,
                name=dotted.rsplit(".", 1)[-1],
                receiver=receiver,
                arg_names=args,
                cell=self.cell,
                line=self._line(node),
                loop_depth=self.loop_depth,
            )
        )

    def _record_assign(self, targets: set[str], value: cst.BaseExpression, node: cst.CSTNode) -> None:
        call = None
        if isinstance(value, cst.Call):
            # the visitor also sees this call via visit_Call; link the same info
            dotted = _dotted(value.func)
            if dotted is not None:
                receiver = _dotted(value.func.value) if isinstance(value.func, cst.Attribute) else None
                args: set[str] = set()
                for arg in value.args:
                    args |= expr_names(arg.value)
                call = CallSite(
                    func=dotted,
                    name=dotted.rsplit(".", 1)[-1],
                    receiver=receiver,
                    arg_names=args,
                    cell=self.cell,
                    line=self._line(node),
                    loop_depth=self.loop_depth,
                )
        self.assigns.append(
            AssignEvent(
                targets=targets,
                sources=expr_names(value),
                cell=self.cell,
                line=self._line(node),
                call=call,
            )
        )

    def visit_Assign(self, node: cst.Assign) -> None:
        targets: set[str] = set()
        for t in node.targets:
            targets |= _target_names(t.target)
        self._record_assign(targets, node.value, node)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if node.value is not None:
            self._record_assign(_target_names(node.target), node.value, node)

    def visit_AugAssign(self, node: cst.AugAssign) -> None:
        targets = _target_names(node.target)
        self._record_assign(targets, node.value, node)


def _target_names(node: cst.BaseExpression) -> set[str]:
    if isinstance(node, cst.Name):
        return {node.value}
    if isinstance(node, (cst.Tuple, cst.List)):
        names: set[str] = set()
        for el in node.elements:
            names |= _target_names(el.value)
        return names
    if isinstance(node, (cst.Subscript, cst.Attribute)):
        # df["x"] = ... mutates df; treat df as (re)assigned
        return _target_names(node.value) if not isinstance(node, cst.Attribute) else _target_names(node.value)
    return set()


def _strip_magics(source: str) -> str:
    lines = []
    for line in source.splitlines():
        if line.lstrip().startswith(("%", "!")):
            lines.append("pass  # magic stripped")
        else:
            lines.append(line)
    return "\n".join(lines)


@dataclass
class NotebookDataflow:
    calls: list[CallSite] = field(default_factory=list)
    assigns: list[AssignEvent] = field(default_factory=list)
    deps: dict[str, set[str]] = field(default_factory=dict)  # name -> names it was derived from
    parse_errors: list[int] = field(default_factory=list)  # cell indices that failed to parse

    def ancestors(self, names: set[str]) -> set[str]:
        """Transitive closure of deps: every name the given names derive from."""
        seen: set[str] = set()
        frontier = set(names)
        while frontier:
            n = frontier.pop()
            if n in seen:
                continue
            seen.add(n)
            frontier |= self.deps.get(n, set()) - seen
        return seen


def analyze(nb: ParsedNotebook) -> NotebookDataflow:
    flow = NotebookDataflow()
    for cell in nb.code_cells:
        try:
            wrapper = MetadataWrapper(cst.parse_module(_strip_magics(cell.source)))
        except cst.ParserSyntaxError:
            flow.parse_errors.append(cell.index)
            continue
        visitor = _CellVisitor(cell.index)
        wrapper.visit(visitor)
        flow.calls.extend(visitor.calls)
        flow.assigns.extend(visitor.assigns)
        for ev in visitor.assigns:
            for t in ev.targets:
                flow.deps.setdefault(t, set()).update(ev.sources - {t})
    return flow
