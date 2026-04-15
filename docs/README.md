# Building Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/) with the [Book Theme](https://sphinx-book-theme.readthedocs.io/en/stable/) for maintaining and generating our documentation.

> **Recommended:** Use the Docker-based build to avoid managing Python dependencies entirely. This is the easiest method on all platforms, especially Windows.

> **Note:** If building natively, we strongly recommend using a Python virtual environment to isolate the required dependencies from your system's global Python environment.

## Current-Version Documentation

This section describes how to build the documentation for the current version of the project.

<details open>
<summary><strong>🐳 Docker — Recommended (Linux & Windows)</strong></summary>

```bash
# 1. From the repo root, build and run the docs service
docker compose -f docs/docker-compose.yml up docs

# 2. Open the current docs
# Linux
xdg-open docs/_build/current/index.html
# Windows
start docs\_build\current\index.html
```

> **Note:** The build output is written directly to `docs/_build/current/` on your host. The `docs` service clears `_build/current/` before each build.
</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
# 1. Navigate to the docs directory and install dependencies
cd docs
pip install -r requirements.txt

# 2. Build the current documentation
make current-docs

# 3. Open the current docs
xdg-open _build/current/index.html
```
</details>

<details>
<summary><strong>Windows</strong></summary>

```batch
:: 1. Navigate to the docs directory and install dependencies
cd docs
pip install -r requirements.txt

:: 2. Build the current documentation
make current-docs

:: 3. Open the current docs
start _build\current\index.html
```
</details>


## Multi-Version Documentation

This section describes how to build the multi-version documentation, which includes previous tags and the main branch.

<details open>
<summary><strong>🐳 Docker — Recommended (Linux & Windows)</strong></summary>

```bash
# 1. From the repo root, build and run the docs-multi service
docker compose -f docs/docker-compose.yml up docs-multi

# 2. Open the multi-version docs
# Linux
xdg-open docs/_build/index.html
# Windows
start docs\_build\index.html
```

> **Note:** The build output is written directly to `docs/_build/` on your host. Multi-version builds require all Git tags to be present locally — run `git fetch --tags` before building.
</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
# 1. Navigate to the docs directory and install dependencies
cd docs
pip install -r requirements.txt

# 2. Build the multi-version documentation
make multi-docs

# 3. Open the multi-version docs
xdg-open _build/index.html
```
</details>

<details>
<summary><strong>Windows</strong></summary>

```batch
:: 1. Navigate to the docs directory and install dependencies
cd docs
pip install -r requirements.txt

:: 2. Build the multi-version documentation
make multi-docs

:: 3. Open the multi-version docs
start _build\index.html
```
</details>
