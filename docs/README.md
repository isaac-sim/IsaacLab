# Building Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/) with the [Book Theme](https://sphinx-book-theme.readthedocs.io/en/stable/) for maintaining and generating our documentation.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) before continuing.
Run all commands below from the repository root. The `--isolated` option creates
a temporary environment for the `test` extra, which combines test and
documentation requirements, leaving the repository's `.venv` unchanged.

## Current-Version Documentation

This section describes how to build the documentation for the current version of the project.

<details open>
<summary><strong>Linux</strong></summary>

```bash
# 1. Build the current documentation
uv run --isolated --extra test -- make -C docs current-docs

# 2. Open the current docs
xdg-open docs/_build/current/index.html
```
</details>

<details> <summary><strong>Windows</strong></summary>

```batch
:: 1. Build the current documentation
uv run --isolated --extra test -- cmd /c docs\make.bat current-docs

:: 2. Open the current docs
start docs\_build\current\index.html
```
</details>


## Multi-Version Documentation

This section describes how to build the multi-version documentation, which includes previous tags and the main branch.

<details open> <summary><strong>Linux</strong></summary>

```bash
# 1. Build the multi-version documentation
uv run --isolated --extra test -- make -C docs multi-docs

# 2. Open the multi-version docs
xdg-open docs/_build/index.html
```
</details>

<details> <summary><strong>Windows</strong></summary>

```batch
:: 1. Build the multi-version documentation
uv run --isolated --extra test -- cmd /c docs\make.bat multi-docs

:: 2. Open the multi-version docs
start docs\_build\index.html
```
</details>
