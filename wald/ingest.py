"""nbformat -> ParsedNotebook: the single input model every layer consumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import nbformat


@dataclass
class Cell:
    index: int  # position in the notebook, counting all cells
    cell_type: str  # "code" | "markdown"
    source: str
    outputs_text: str = ""  # concatenated text of stored outputs (code cells)


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


def parse_notebook(path: str | Path) -> ParsedNotebook:
    nb = nbformat.read(str(path), as_version=4)
    return from_nbnode(nb, path=Path(path))


def from_nbnode(nb, path: Path | None = None) -> ParsedNotebook:
    cells = []
    for i, c in enumerate(nb.cells):
        outputs = ""
        if c.cell_type == "code":
            outputs = "\n".join(_output_text(o) for o in c.get("outputs", []))
        cells.append(Cell(index=i, cell_type=c.cell_type, source=c.source, outputs_text=outputs))
    return ParsedNotebook(path=path, cells=cells)
