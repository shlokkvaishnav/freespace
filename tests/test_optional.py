"""Optional dependencies: the package works without `anthropic`, and --vlm claude says how to get it."""
import subprocess
import sys

import pytest


def test_core_imports_without_anthropic():
    # a fresh interpreter in which `import anthropic` fails, as if it were not installed
    code = ("import sys; sys.modules['anthropic'] = None\n"
            "import freespace.cli, freespace.pipeline, freespace.describe\n"
            "assert 'anthropic' not in [m for m in sys.modules if sys.modules[m] is not None]\n")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_claude_labeller_explains_missing_package(monkeypatch):
    from freespace import describe

    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError, match=r"\[claude\]"):
        describe.require_anthropic()
