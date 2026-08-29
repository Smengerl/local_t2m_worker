"""Opt-2 — scripts/run.sh must not advertise flags cli.py doesn't accept.

run.sh forwards unknown flags straight to generate.py, where argparse rejects
them under `set -e`, so a wrong name in the usage text is a broken example.
"""

import ast
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# flags run.sh handles itself and never forwards to generate.py
_SHELL_ONLY = {"--queue", "--offline", "--help"}


def _cli_long_flags() -> set[str]:
    tree = ast.parse((_ROOT / "cli.py").read_text())
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                    flags.add(arg.value)
    return flags


def _run_sh_usage_flags() -> set[str]:
    text = (_ROOT / "scripts" / "run.sh").read_text()
    usage = text.split("usage() {", 1)[1].split("EOF", 1)[0]
    return set(re.findall(r"--[a-z][a-z-]+", usage))


def test_run_sh_only_documents_real_flags():
    known = _cli_long_flags() | _SHELL_ONLY
    documented = _run_sh_usage_flags()
    unknown = documented - known
    assert not unknown, f"run.sh usage documents unknown flags: {sorted(unknown)}"


def test_cli_flag_extraction_sane():
    # guard against the ast walk silently returning nothing
    assert {"--config", "--cfg-scale", "--model-repo"} <= _cli_long_flags()
