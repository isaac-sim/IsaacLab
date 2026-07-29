# Building Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/) with the [Book Theme](https://sphinx-book-theme.readthedocs.io/en/stable/) for maintaining and generating our documentation.

> **Note:** To avoid dependency conflicts, we strongly recommend using a Python virtual environment to isolate the required dependencies from your system's global Python environment.

## Current-Version Documentation

This section describes how to build the documentation for the current version of the project.

<details open>
<summary><strong>Linux</strong></summary>

```bash
# 1. From the repository root, install the locked docs dependency group
uv sync --locked --only-group docs

# 2. Activate the environment
source .venv/bin/activate

# 3. Navigate to the docs directory
cd docs

# 4. Build the current documentation
make current-docs

# 5. Open the current docs
xdg-open _build/current/index.html
```
</details>

<details> <summary><strong>Windows</strong></summary>

```batch
:: 1. From the repository root, install the locked docs dependency group
uv sync --locked --only-group docs

:: 2. Activate the environment
.venv\Scripts\activate.bat

:: 3. Navigate to the docs directory
cd docs

:: 4. Build the current documentation
make current-docs

:: 5. Open the current docs
start _build\current\index.html
```
</details>


## Multi-Version Documentation

This section describes how to build the multi-version documentation, which includes previous tags and the main branch.

<details open> <summary><strong>Linux</strong></summary>

```bash
# 1. From the repository root, install the locked docs dependency group
uv sync --locked --only-group docs

# 2. Activate the environment
source .venv/bin/activate

# 3. Navigate to the docs directory
cd docs

# 4. Build the multi-version documentation
make multi-docs

# 5. Open the multi-version docs
xdg-open _build/index.html
```
</details>

<details> <summary><strong>Windows</strong></summary>

```batch
:: 1. From the repository root, install the locked docs dependency group
uv sync --locked --only-group docs

:: 2. Activate the environment
.venv\Scripts\activate.bat

:: 3. Navigate to the docs directory
cd docs

:: 4. Build the multi-version documentation
make multi-docs

:: 5. Open the multi-version docs
start _build\index.html
```
</details>
