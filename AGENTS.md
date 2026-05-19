# AGENTS.md

Guidelines for contributors and AI agents working on this codebase.

## Tooling

- Use `uv` for all commands: `uv run pytest`, `uv run python -m mlcast`, etc.
- Linting and formatting: `ruff` (configured in `pyproject.toml`).
- Pre-commit hooks are installed via `uv run pre-commit install`. Always run and
  pass hooks before proposing a commit.
- **Never commit unless explicitly asked.** When asked, commit in logical groups
  with clear, descriptive messages following the existing `type: description`
  convention (e.g. `feat:`, `fix:`, `refactor:`, `test:`).

## Configuration

- All runtime configuration goes through [Fiddle](https://github.com/google/fiddle)
  (`fdl`). Config changes must go through fiddlers or `set:` overrides — never
  mutate config objects outside of a fiddler or a test setup.
- No inline imports. All imports belong at the top of the file.

## Logging

- Use `loguru` exclusively. Do not use the stdlib `logging` module.

## Code style

- Docstrings follow NumPy style.
- Add inline comments on any non-obvious Fiddle internals (e.g. direct access to
  private `FiddleFlag` attributes).
- Use `jaxtyping` annotations on `torch.nn.Module` classes to document tensor
  shapes and dtypes.

## Tests

- Integration (CLI) tests invoke `python -m mlcast` via `subprocess.run` so that
  absl-py flag state is fully isolated between test runs.
- Unit tests that would instantiate heavy PyTorch objects use
  `unittest.mock.patch` to avoid it (see `tests/config/test_orchestrator.py`).
- All test functions carry full type annotations on their parameters.

### Fixture naming conventions

| Prefix | Type | Meaning |
|--------|------|---------|
| `fp_`  | `Path` | Filesystem path to a file or directory |

Example: `fp_test_dataset: Path` — the local path to the cached test zarr store.
Fixtures return paths rather than open dataset handles so that each test (and
each dataloader worker subprocess) can open the resource independently.
