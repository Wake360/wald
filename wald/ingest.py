"""nbformat -> ParsedNotebook: the single input model every layer consumes."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
from nbformat.warnings import MissingIDFieldWarning


@dataclass
class Cell:
    index: int  # position in the notebook, counting all cells
    cell_type: str  # "code" | "markdown"
    source: str
    outputs_text: str = ""  # concatenated text of stored outputs (code cells)
    start_line: int = 1  # 1-indexed file line of the cell's first source line (scripts)


@dataclass
class ParsedNotebook:
    path: Path | None
    cells: list[Cell] = field(default_factory=list)

    @property
    def code_cells(self) -> list[Cell]:
        return [c for c in self.cells if c.cell_type == "code"]

    @property
    def markdown_cells(self) -> list[Cell]:
        return [c for c in self.cells if c.cell_type == "markdown"]

    def full_source(self) -> str:
        return "\n".join(c.source for c in self.code_cells)


def _output_text(output) -> str:
    if output.get("output_type") == "stream":
        return output.get("text", "")
    data = output.get("data", {})
    return data.get("text/plain", "")


def _coerce_source(value) -> str:
    """nbformat stores source as a str, but hand-edited/lossy JSON can carry a
    line list or an explicit null; both must lint, not crash downstream."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(str(x) for x in value)
    return str(value)


MAX_NOTEBOOK_BYTES = 20 * 1024 * 1024
MAX_NOTEBOOK_CELLS = 5000


def parse_notebook(path: str | Path) -> ParsedNotebook:
    if Path(path).suffix == ".py":
        return parse_script(path)
    if Path(path).is_file() and Path(path).stat().st_size > MAX_NOTEBOOK_BYTES:
        raise ValueError("input exceeds 20 MB cap")
    with warnings.catch_warnings():
        # valid older notebooks lack per-cell `id`; nbformat's read-time
        # validate() warns about it, which is not a wald-level problem
        warnings.simplefilter("ignore", MissingIDFieldWarning)
        try:
            nb = nbformat.read(str(path), as_version=4)
        except TypeError as exc:
            # e.g. "cells": null reaches rejoin_lines and iterates None;
            # surface as ValueError so the CLI maps it to a clean exit 3
            raise ValueError("not a valid notebook (malformed structure)") from exc
    if len(nb.cells) > MAX_NOTEBOOK_CELLS:
        raise ValueError("input exceeds 5000-cell cap")
    return from_nbnode(nb, path=Path(path))


_PERCENT_MARKER = re.compile(r"^#\s*%%")


def _strip_comment(line: str) -> str:
    if line.startswith("# "):
        return line[2:]
    if line.startswith("#"):
        return line[1:]
    return line


def parse_script(path: str | Path) -> ParsedNotebook:
    """Percent-format (`# %%`) or plain .py script -> ParsedNotebook. A file
    with no markers is one code cell; markers split cells, and a `[markdown]`
    marker turns the following comment block into a markdown cell. start_line
    is the 1-indexed file line of each cell's first source line."""
    p = Path(path)
    if p.is_file() and p.stat().st_size > MAX_NOTEBOOK_BYTES:
        raise ValueError("input exceeds 20 MB cap")
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    markers = [i for i, ln in enumerate(lines) if _PERCENT_MARKER.match(ln)]
    if not markers:
        return ParsedNotebook(path=p, cells=[
            Cell(index=0, cell_type="code", source=text, start_line=1),
        ])
    cells: list[Cell] = []
    if markers[0] > 0 and any(ln.strip() for ln in lines[:markers[0]]):
        cells.append(Cell(index=0, cell_type="code",
                          source="\n".join(lines[:markers[0]]), start_line=1))
    for j, m in enumerate(markers):
        end = markers[j + 1] if j + 1 < len(markers) else len(lines)
        body = lines[m + 1:end]
        if "[markdown]" in lines[m]:
            cells.append(Cell(index=len(cells), cell_type="markdown",
                              source="\n".join(_strip_comment(ln) for ln in body),
                              start_line=m + 2))
        else:
            cells.append(Cell(index=len(cells), cell_type="code",
                              source="\n".join(body), start_line=m + 2))
    if len(cells) > MAX_NOTEBOOK_CELLS:
        raise ValueError("input exceeds 5000-cell cap")
    return ParsedNotebook(path=p, cells=cells)


def from_nbnode(nb, path: Path | None = None) -> ParsedNotebook:
    cells = []
    for i, c in enumerate(nb.cells):
        cell_type = c.get("cell_type", "")
        outputs = ""
        if cell_type == "code":
            raw = c.get("outputs") or []  # missing or explicit-null outputs -> []
            if not isinstance(raw, list):
                raw = []
            outputs = "\n".join(_output_text(o) for o in raw)
        cells.append(Cell(
            index=i,
            cell_type=cell_type,
            source=_coerce_source(c.get("source")),
            outputs_text=outputs,
        ))
    return ParsedNotebook(path=path, cells=cells)
