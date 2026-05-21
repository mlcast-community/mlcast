"""Integration tests that execute the Python snippets embedded in README.md.

Each ```python block in the README is extracted, checked for runnability
(snippets without import statements are continuations and are skipped), and
executed in an isolated namespace with lightweight training overrides injected
immediately before the terminal training call.
"""

import ast
from pathlib import Path
from typing import Any

import fiddle as fdl
import pytest

from mlcast.config.fiddlers import set_variables, use_random_sampler

_README = Path(__file__).parent.parent / "README.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_python_snippets(readme: Path) -> list[str]:
    """Parse *readme* line-by-line and return all ```python code blocks.

    Parameters
    ----------
    readme : Path
        Path to the Markdown file.

    Returns
    -------
    list of str
        Each element is the raw source of one ```python … ``` block.
    """
    snippets: list[str] = []
    current: list[str] = []
    in_block = False
    for line in readme.read_text().splitlines():
        if line.strip() == "```python":
            in_block = True
            current = []
        elif line.strip() == "```" and in_block:
            in_block = False
            snippets.append("\n".join(current))
        elif in_block:
            current.append(line)
    return snippets


def _is_continuation(snippet: str) -> bool:
    """Return True if *snippet* cannot be executed standalone.

    A snippet is treated as a continuation when it either has no import
    statements at all, or references ``cfg`` without ever assigning it (i.e.
    ``cfg`` is a free variable — the snippet relies on a previous block having
    defined it).

    Parameters
    ----------
    snippet : str
        Python source code to inspect.

    Returns
    -------
    bool
    """
    try:
        tree = ast.parse(snippet)
    except SyntaxError:
        return True

    has_imports = any(isinstance(node, ast.Import | ast.ImportFrom) for node in ast.walk(tree))
    if not has_imports:
        return True

    # Check whether cfg is assigned anywhere in the snippet.
    cfg_assigned = any(
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cfg" for t in node.targets)
        for node in ast.walk(tree)
    )
    cfg_referenced = any(isinstance(node, ast.Name) and node.id == "cfg" for node in ast.walk(tree))
    # If cfg is used but never assigned, this is a continuation block.
    if cfg_referenced and not cfg_assigned:
        return True

    return False


def _requires_mfai(snippet: str) -> bool:
    """Return True if *snippet* imports from the ``mfai`` package.

    Parameters
    ----------
    snippet : str
        Python source code to inspect.

    Returns
    -------
    bool
    """
    return "from mfai" in snippet or "import mfai" in snippet


def _patch_cfg(cfg: fdl.Config, fp_dataset: Path, tmp_path: Path) -> None:
    """Override heavy training settings so the snippet runs in seconds.

    Uses the ``set_variables`` fiddler (rather than direct assignment) so that
    ``network.input_channels`` is kept in sync with ``standard_names``.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration graph to mutate in-place.
    fp_dataset : Path
        Local path to the cached test zarr store.
    tmp_path : Path
        Pytest-provided temporary directory for trainer outputs.
    """
    cfg.data.dataset_factory.zarr_path = str(fp_dataset.absolute())
    set_variables(cfg, standard_names=["rainfall_flux"])
    # Switch to the on-the-fly random sampler so no pre-computed CSV is needed.
    use_random_sampler(cfg)
    cfg.data.splits = {"time": {"train": 0.4, "val": 0.3, "test": 0.3}}
    cfg.trainer.fast_dev_run = True
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    cfg.trainer.default_root_dir = str(tmp_path)


def _inject_patch(snippet: str) -> ast.Module:
    """AST-transform *snippet* to call ``_patch_cfg`` before the training call.

    Finds the last top-level ``ast.Expr`` whose value is a call to
    ``train_from_config`` or ``experiment.run``, and inserts a call to
    ``_patch_cfg(cfg, fp_test_dataset, tmp_path)`` immediately before it.

    Parameters
    ----------
    snippet : str
        Python source of a README snippet.

    Returns
    -------
    ast.Module
        The transformed (and location-fixed) AST module.

    Raises
    ------
    ValueError
        If no terminal training call is found in the snippet.
    """
    tree = ast.parse(snippet)

    def _is_training_call(node: ast.stmt) -> bool:
        if not isinstance(node, ast.Expr):
            return False
        call = node.value
        if not isinstance(call, ast.Call):
            return False
        func = call.func
        # train_from_config(cfg)
        if isinstance(func, ast.Name) and func.id == "train_from_config":
            return True
        # experiment.run()
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "experiment"
        ):
            return True
        return False

    # Find the index of the last training call in the top-level body.
    target_idx: int | None = None
    for i, node in enumerate(tree.body):
        if _is_training_call(node):
            target_idx = i

    if target_idx is None:
        raise ValueError(f"No terminal train_from_config(...) or experiment.run() call found in snippet:\n{snippet}")

    # Build: _patch_cfg(cfg, fp_test_dataset, tmp_path)
    patch_call = ast.Expr(
        value=ast.Call(
            func=ast.Name(id="_patch_cfg", ctx=ast.Load()),
            args=[
                ast.Name(id="cfg", ctx=ast.Load()),
                ast.Name(id="fp_test_dataset", ctx=ast.Load()),
                ast.Name(id="tmp_path", ctx=ast.Load()),
            ],
            keywords=[],
        )
    )

    tree.body.insert(target_idx, patch_call)
    return ast.fix_missing_locations(tree)


def _collect_runnable_snippets() -> list[str]:
    """Extract and filter the README Python snippets at collection time.

    Returns
    -------
    list of str
        Snippets that have at least one import statement and can be run
        standalone (continuation blocks are excluded).
    """
    return [s for s in _extract_python_snippets(_README) if not _is_continuation(s)]


# Collected once at module import time so pytest parametrize can use it.
_RUNNABLE_SNIPPETS = _collect_runnable_snippets()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("snippet", _RUNNABLE_SNIPPETS)
def test_readme_snippet(snippet: str, fp_test_dataset: Path, tmp_path: Path) -> None:
    """Execute a README Python snippet with lightweight training overrides.

    Parameters
    ----------
    snippet : str
        Raw Python source extracted from a ```python block in README.md.
    fp_test_dataset : Path
        Fixture providing the local path to the cached test zarr store.
    tmp_path : Path
        Pytest-provided per-test temporary directory.
    """
    if _requires_mfai(snippet):
        pytest.importorskip("mfai")

    tree = _inject_patch(snippet)
    code = compile(tree, filename="<readme_snippet>", mode="exec")
    namespace: dict[str, Any] = {
        "_patch_cfg": _patch_cfg,
        "fp_test_dataset": fp_test_dataset,
        "tmp_path": tmp_path,
    }
    exec(code, namespace)  # noqa: S102
