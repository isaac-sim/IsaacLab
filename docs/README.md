# Building Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/) with the [Book Theme](https://sphinx-book-theme.readthedocs.io/en/stable/) for maintaining and generating our documentation.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) before continuing.
Run the build commands below from the repository root. The Isaac Lab CLI runs
Sphinx in an isolated uv environment with the `dev` extra. The project selects
Python 3.12.

## Current-Version Documentation

This section describes how to build the documentation for the current version of the project.

```bash
uv run isaaclab --docs
```

Open `docs/_build/current/index.html` in a browser after the build completes.

## Multi-Version Documentation

This section describes how to build the multi-version documentation, which includes previous tags and the main branch.

```bash
uv run isaaclab --docs-multi
```

This build requires the Git tags for the versions to include. It checks for
`v3.0.0-EA/index.html` by default; set `DOCS_DEFAULT_REF` to another built ref
if needed. Open `docs/_build/index.html` in a browser after the build completes.
