"""interp-engine is the interpretability runtime; this project must not grow a second one.

These are not style checks. Every mechanism listed here is one interp-engine already provides,
standardised across architectures and validated against TransformerLens and nnsight. A bespoke
copy in this repository would be unvalidated, model-specific, and silently wrong on exactly the
architectures (post-norm Gemma, multi-stream trunks) where the distinction matters.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "microscope"

FORBIDDEN_CALLS = {
    "register_forward_hook",
    "register_forward_pre_hook",
    "register_full_backward_hook",
    "register_module_forward_hook",
}


def _python_files():
    return sorted(SRC.glob("*.py"))


def test_no_bespoke_pytorch_hooks_anywhere_in_the_project():
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_CALLS:
                raise AssertionError(
                    f"{path.name} calls {node.attr}. interp-engine owns activation capture and "
                    "writing; use its points and SteeringSpec instead."
                )


def test_interp_engine_is_only_imported_from_the_adapter():
    """One file talks to the engine, so its API surface is auditable in one place."""
    importers = []
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("interp_engine"):
                importers.append(path.name)
            if isinstance(node, ast.Import) and any(a.name.startswith("interp_engine") for a in node.names):
                importers.append(path.name)
    assert set(importers) == {"interp.py"}, f"interp_engine imported outside the adapter: {sorted(set(importers))}"


def test_the_adapter_only_uses_the_engine_public_namespace():
    """Submodule imports are explicitly not API (interp-engine AGENT_INTEGRATION.md)."""
    tree = ast.parse((SRC / "interp.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("interp_engine."):
            raise AssertionError(
                f"interp.py imports the submodule {node.module}. Only the top-level "
                "interp_engine namespace is API."
            )
