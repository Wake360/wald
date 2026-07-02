"""Notebook execution for corpus building and mutation verification.

Only trusted, self-authored corpus notebooks are executed here. User
notebooks are never executed unless they pass --execute explicitly (M2+).
"""

from __future__ import annotations

import copy

import nbformat
from nbclient import NotebookClient


def execute(nb_node, timeout: int = 180):
    """Execute a notebook node in place-safe copy; returns the executed node."""
    nb = copy.deepcopy(nb_node)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    client.execute()
    return nb


def with_appended_code_cell(nb_node, source: str):
    nb = copy.deepcopy(nb_node)
    nb.cells.append(nbformat.v4.new_code_cell(source))
    return nb


def stdout_lines(nb_node, prefix: str) -> list[str]:
    """Collect stream-output lines starting with prefix across all cells."""
    lines = []
    for cell in nb_node.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                for line in out.get("text", "").splitlines():
                    if line.startswith(prefix):
                        lines.append(line)
    return lines
