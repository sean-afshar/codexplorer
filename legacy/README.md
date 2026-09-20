# Pre-merge packages

`shayan/` and `cex/` are the two packages that `connexplorer` was merged from,
kept for reference together with their tests, notebooks and the cex data-layout
note. They are not installed and not tested by CI. To run them anyway:

```bash
PYTHONPATH=legacy uv run python -c "import shayan, cex"
PYTHONPATH=legacy uv run pytest legacy/tests
```

Both emit a `DeprecationWarning` on import. The method-by-method mapping to the
merged package is in `docs/migrating_from_shayan.md` and `docs/migrating_from_cex.md`.
