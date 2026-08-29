"""Opt-1 — no statements after a function-level `return` (unreachable code)."""

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PY_FILES = sorted(
    p for p in (_ROOT / "pipelines").glob("*.py")
) + [_ROOT / "generate.py"]


class _UnreachableFinder(ast.NodeVisitor):
    def __init__(self):
        self.hits: list[tuple[str, int]] = []

    def _check_body(self, body, fn_name):
        for i, stmt in enumerate(body[:-1]):
            if isinstance(stmt, (ast.Return, ast.Raise)):
                nxt = body[i + 1]
                self.hits.append((fn_name, nxt.lineno))

    def visit_FunctionDef(self, node):
        self._check_body(node.body, node.name)
        self.generic_visit(node)


@pytest.mark.parametrize("path", _PY_FILES, ids=lambda p: p.name)
def test_no_unreachable_statements(path):
    finder = _UnreachableFinder()
    finder.visit(ast.parse(path.read_text()))
    assert finder.hits == [], f"unreachable code in {path.name}: {finder.hits}"
