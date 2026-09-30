Added
^^^^^

* Added a cross-platform CLI command for building multi-version documentation with uv using ``isaaclab --docs_multi``.

Fixed
^^^^^

* Fixed the Windows documentation instructions to use the uv-backed CLI instead of the batch build script.
* Reused the repository environment for documentation dependencies to avoid a second full installation.
* Cleared current documentation output and its Sphinx cache before building to avoid stale pages and warnings.
* Reported a missing multi-version redirect target as a CLI error with recovery guidance.
