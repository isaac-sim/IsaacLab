# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation workflow training tests.

Covers all documented installation methods from the IsaacLab installation guide
(``docs/source/setup/installation/``):

+-----------------------------------+--------------+-------------------+------------+
| Test class                        | Env manager  | Isaac Sim source  | Doc ref    |
+===================================+==============+===================+============+
| Test_UV_Kitless_Training          | uv           | none              | kitless    |
| Test_UV_IsaacSim_Pip_Training     | uv           | PyPI              | pip        |
| Test_UV_IsaacSim_Source_With_Env  | uv           | _isaac_sim        | binaries   |
| Test_Conda_Kitless_Training       | conda        | none              | kitless    |
| Test_Conda_IsaacSim_Pip_Training  | conda        | PyPI              | pip        |
| Test_Conda_IsaacSim_Source        | conda        | _isaac_sim        | binaries   |
| Test_UV_IsaacSim_Source_Training  | none (symlink| _isaac_sim        | binaries   |
|                                   | python.sh)   |                   | (no-env)   |
+-----------------------------------+--------------+-------------------+------------+

Each test:
  1. Creates a clean environment (uv venv, conda env, or none for source).
  2. Installs the documented package set via ``./isaaclab.sh -i …``.
  3. Runs a short Cartpole training smoke via ``rsl_rl/train.py`` (3 iterations).
  4. Asserts exit code 0 and no Python traceback in combined stdout/stderr.
  5. Tears down the environment unconditionally (where applicable).

Design note — selective vs default (all-mode) install for kitless:
  The documented quick-start says ``./isaaclab.sh --install`` (all-mode, no args).
  The training tests deliberately use a *selective* install
  (``newton,tasks,assets,rl[rsl_rl]``) because:
  (a) it is faster; (b) it precisely reflects the Newton-only subset documented
  in the ``kitless_installation.rst`` Selective Install table; (c) the
  all-mode install would also pull in ``isaaclab_physx``/``isaaclab_ov`` which
  are unused without Isaac Sim and would add noise to a kitless smoke test.
  The ``newton`` token auto-adds ``visualizers[newton]`` (see ``install.py``).

Env vars that control Isaac Sim pip installs (see ``install.py`` for full docs):
  ISAACSIM_VERSION_SPEC   version specifier, e.g. ``"==6.4.0"``
  ISAACSIM_EXTRAS         pip extras, e.g. ``"all,extscache"`` (default: ``"all"``)
  ISAACSIM_EXTRA_INDEX_URLS  space-separated extra index URLs; triggers the two
                           internal NVIDIA Artifactory registries automatically
  ISAACSIM_USE_PRE        set to ``"1"`` to pass ``--pre`` (for pre-release builds)

CI invocation:
  # uv image (Dockerfile.installci):
  tools/run_install_ci.py docker --gpu -- -sv -m uv --tb=short

  # conda image (Dockerfile.installci-conda):
  tools/run_install_ci.py docker --conda --gpu -- -sv -m conda --tb=short

  # Isaac Sim base image with _isaac_sim (source-build or binaries):
  tools/run_install_ci.py docker --gpu -- -sv -m isaacsim_source --tb=short
"""

from __future__ import annotations

import os
import shutil

import pytest
from utils import Conda_Mixin, IsaacSimSource_Mixin, UV_Mixin, find_isaaclab_root

# ---------------------------------------------------------------------------
# Shared install strings
# ---------------------------------------------------------------------------

# The `newton` token auto-includes `visualizers[newton]` (see install.py),
# so listing it explicitly here would be redundant.
_KITLESS_INSTALL = "newton,tasks,assets,rl[rsl_rl]"

# Used when pip-installing Isaac Sim (public pypi.nvidia.com or internal registry).
_ISAACSIM_PIP_INSTALL = "isaacsim,physx,tasks,assets,rl[rsl_rl]"

# Used when Isaac Sim is already present via _isaac_sim symlink; skip the pip step.
_ISAACSIM_SOURCE_INSTALL = "physx,tasks,assets,rl[rsl_rl]"

# ---------------------------------------------------------------------------
# Shared training commands
# ---------------------------------------------------------------------------

_KITLESS_TRAIN_CMD = [
    "scripts/reinforcement_learning/rsl_rl/train.py",
    "--task",
    "Isaac-Cartpole-Direct-v0",
    "--num_envs",
    "16",
    "--max_iterations",
    "3",
    "presets=newton_mjwarp",
    "--visualizer",
    "newton",
]

_ISAACSIM_TRAIN_CMD = [
    "scripts/reinforcement_learning/rsl_rl/train.py",
    "--task",
    "Isaac-Cartpole-Direct-v0",
    "--num_envs",
    "16",
    "--max_iterations",
    "3",
    "presets=physx",
    "--headless",
]

# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------


def _assert_no_crash(result, label: str) -> None:
    """Assert that *result* has exit code 0 and no Python traceback."""
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{label} failed (rc={result.returncode}):\n{output}"
    assert "Traceback (most recent call last):" not in output, f"{label} produced a Python traceback:\n{output}"


def _isaacsim_pip_available() -> bool:
    """Return True when Isaac Sim can be obtained via pip.

    Considers two signals:
    * ``ISAACSIM_EXTRA_INDEX_URLS`` is set → an internal registry is configured,
      assume the build is available there.
    * ``import isaacsim`` succeeds in the host Python → it is already installed
      at the system level (e.g. in the base Docker image).
    """
    if os.environ.get("ISAACSIM_EXTRA_INDEX_URLS"):
        return True
    try:
        import isaacsim  # noqa: F401

        return True
    except ImportError:
        pass
    return False


def _isaacsim_source_available() -> bool:
    """Return True when the ``_isaac_sim`` symlink is present in the repo root."""
    return (find_isaaclab_root() / "_isaac_sim").exists()


# ---------------------------------------------------------------------------
# uv + kitless Newton
# ---------------------------------------------------------------------------


class Test_UV_Kitless_Training(UV_Mixin):
    """Install via uv, train on Newton (kitless) backend."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(1800)
    def test_install_and_train_newton(self, isaaclab_root):
        """``isaaclab.sh -i newton,…`` then train Isaac-Cartpole-Direct-v0 with Newton."""
        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", _KITLESS_INSTALL], cwd=isaaclab_root)
            _assert_no_crash(result, "isaaclab -i (newton, uv)")

            result = self.run_in_uv_env([str(self.cli_script), "-p"] + _KITLESS_TRAIN_CMD, cwd=isaaclab_root)
            _assert_no_crash(result, "train.py newton (uv)")
        finally:
            self.destroy_uv_env()


# ---------------------------------------------------------------------------
# uv + Isaac Sim / PhysX — pip install (public or internal registry)
# ---------------------------------------------------------------------------


class Test_UV_IsaacSim_Pip_Training(UV_Mixin):
    """Install Isaac Sim via pip into a uv venv, train on PhysX backend.

    Skipped unless ``import isaacsim`` succeeds in the host Python or
    ``ISAACSIM_EXTRA_INDEX_URLS`` points to an internal registry.
    Set ``ISAACSIM_VERSION_SPEC``, ``ISAACSIM_USE_PRE``, and/or
    ``ISAACSIM_EXTRA_INDEX_URLS`` to target an internal pre-release build::

        ISAACSIM_VERSION_SPEC="==6.4.0" \\
        ISAACSIM_USE_PRE=1 \\
        ISAACSIM_EXTRAS="all,extscache" \\
        ISAACSIM_EXTRA_INDEX_URLS="https://urm.nvidia.com/artifactory/api/pypi/sw-isaacsim-pypi/simple" \\
        tools/run_install_ci.py docker --gpu -- -sv -m "uv and slow"
    """

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")
        if not _isaacsim_pip_available():
            pytest.skip("isaacsim not importable and ISAACSIM_EXTRA_INDEX_URLS not set; skip pip-based Isaac Sim test")

    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(1800)
    def test_install_and_train_physx(self, isaaclab_root):
        """``isaaclab.sh -i isaacsim,physx,…`` then train Isaac-Cartpole-Direct-v0 with PhysX."""
        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", _ISAACSIM_PIP_INSTALL], cwd=isaaclab_root)
            _assert_no_crash(result, "isaaclab -i (isaacsim/physx pip, uv)")

            result = self.run_in_uv_env([str(self.cli_script), "-p"] + _ISAACSIM_TRAIN_CMD, cwd=isaaclab_root)
            _assert_no_crash(result, "train.py physx (pip, uv)")
        finally:
            self.destroy_uv_env()


# ---------------------------------------------------------------------------
# uv context + Isaac Sim / PhysX — source build (_isaac_sim symlink)
# ---------------------------------------------------------------------------


class Test_UV_IsaacSim_Source_Training(IsaacSimSource_Mixin):
    """Isaac Sim is pre-installed via ``_isaac_sim`` symlink; install physx/tasks only.

    No fresh venv is created — ``isaaclab.sh`` is invoked with ``VIRTUAL_ENV``
    and ``CONDA_PREFIX`` unset so it falls through to ``_isaac_sim/python.sh``.

    Skipped when the ``_isaac_sim`` symlink is absent from the repo root.
    """

    @classmethod
    def setup_class(cls):
        if not _isaacsim_source_available():
            pytest.skip("_isaac_sim symlink not found; skip source-build Isaac Sim test")

    @pytest.mark.isaacsim_source
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(1800)
    def test_install_and_train_physx(self, isaaclab_root):
        """``isaaclab.sh -i physx,…`` (no isaacsim token) then train with PhysX."""
        cli = str(isaaclab_root / "isaaclab.sh")

        result = self.run_without_venv([cli, "-i", _ISAACSIM_SOURCE_INSTALL], cwd=isaaclab_root)
        _assert_no_crash(result, "isaaclab -i (physx source, no venv)")

        result = self.run_without_venv([cli, "-p"] + _ISAACSIM_TRAIN_CMD, cwd=isaaclab_root)
        _assert_no_crash(result, "train.py physx (source, no venv)")


# ---------------------------------------------------------------------------
# uv env + Isaac Sim / PhysX — source build (_isaac_sim symlink)
# Documented in: binaries_installation.rst, source_installation.rst
#   "Setting up a Python Environment → UV Environment"
# ---------------------------------------------------------------------------


class Test_UV_IsaacSim_Source_With_Env_Training(UV_Mixin, IsaacSimSource_Mixin):
    """_isaac_sim symlink + fresh uv venv; models the documented binaries/source install.

    This is the **uv env variant** of the binaries or source-build workflow:

    .. code-block:: bash

       ln -s ${ISAACSIM_PATH} _isaac_sim      # (already done by the CI image)
       ./isaaclab.sh --uv env_isaaclab         # create a uv venv
       source env_isaaclab/bin/activate
       ./isaaclab.sh --install                 # install Isaac Lab into the venv

    ``isaaclab.sh`` uses the **venv python** (``VIRTUAL_ENV`` is set) and
    *also* sources ``_isaac_sim/setup_conda_env.sh`` (lines 34–36 of
    ``isaaclab.sh``), which puts Isaac Sim's packages onto ``PYTHONPATH`` even
    without an explicit pip install of ``isaacsim``.

    Skipped when ``_isaac_sim`` symlink is absent or ``uv`` is not on ``PATH``.
    """

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")
        if not _isaacsim_source_available():
            pytest.skip("_isaac_sim symlink not found; skip binaries+uv-env source test")

    @pytest.mark.uv
    @pytest.mark.isaacsim_source
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(1800)
    def test_install_and_train_physx(self, isaaclab_root):
        """Create uv venv, install ``physx,…`` (no isaacsim pip), train PhysX via PYTHONPATH."""
        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", _ISAACSIM_SOURCE_INSTALL], cwd=isaaclab_root)
            _assert_no_crash(result, "isaaclab -i (physx source, uv env)")

            result = self.run_in_uv_env([str(self.cli_script), "-p"] + _ISAACSIM_TRAIN_CMD, cwd=isaaclab_root)
            _assert_no_crash(result, "train.py physx (source, uv env)")
        finally:
            self.destroy_uv_env()


# ---------------------------------------------------------------------------
# conda + kitless Newton
# ---------------------------------------------------------------------------


class Test_Conda_Kitless_Training(Conda_Mixin):
    """Install via conda, train on Newton (kitless) backend."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("conda"):
            pytest.skip("conda is not available")

    @pytest.mark.conda
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(1800)
    def test_install_and_train_newton(self, isaaclab_root):
        """``isaaclab.sh -c`` + ``isaaclab.sh -i newton,…`` then train Newton."""
        cli = str(isaaclab_root / "isaaclab.sh")
        try:
            self.create_conda_env(isaaclab_root)

            result = self.run_in_conda_env([cli, "-i", _KITLESS_INSTALL], cwd=isaaclab_root)
            _assert_no_crash(result, "isaaclab -i (newton, conda)")

            result = self.run_in_conda_env([cli, "-p"] + _KITLESS_TRAIN_CMD, cwd=isaaclab_root)
            _assert_no_crash(result, "train.py newton (conda)")
        finally:
            self.destroy_conda_env()


# ---------------------------------------------------------------------------
# conda + Isaac Sim / PhysX — pip install (public or internal registry)
# ---------------------------------------------------------------------------


class Test_Conda_IsaacSim_Pip_Training(Conda_Mixin):
    """Install Isaac Sim via pip inside a conda env, train on PhysX backend.

    See ``Test_UV_IsaacSim_Pip_Training`` for the env-var interface.
    """

    @classmethod
    def setup_class(cls):
        if not shutil.which("conda"):
            pytest.skip("conda is not available")
        if not _isaacsim_pip_available():
            pytest.skip("isaacsim not importable and ISAACSIM_EXTRA_INDEX_URLS not set; skip pip-based Isaac Sim test")

    @pytest.mark.conda
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(1800)
    def test_install_and_train_physx(self, isaaclab_root):
        """``isaaclab.sh -c`` + ``isaaclab.sh -i isaacsim,physx,…`` then train PhysX."""
        cli = str(isaaclab_root / "isaaclab.sh")
        try:
            self.create_conda_env(isaaclab_root)

            result = self.run_in_conda_env([cli, "-i", _ISAACSIM_PIP_INSTALL], cwd=isaaclab_root)
            _assert_no_crash(result, "isaaclab -i (isaacsim/physx pip, conda)")

            result = self.run_in_conda_env([cli, "-p"] + _ISAACSIM_TRAIN_CMD, cwd=isaaclab_root)
            _assert_no_crash(result, "train.py physx (pip, conda)")
        finally:
            self.destroy_conda_env()


# ---------------------------------------------------------------------------
# conda + Isaac Sim / PhysX — source build (_isaac_sim symlink)
# ---------------------------------------------------------------------------


class Test_Conda_IsaacSim_Source_Training(Conda_Mixin, IsaacSimSource_Mixin):
    """Isaac Sim is pre-installed via ``_isaac_sim`` symlink; install into a fresh conda env.

    ``./isaaclab.sh -c`` writes activation hooks that source
    ``_isaac_sim/setup_conda_env.sh``, so the symlink's packages are on
    ``PYTHONPATH`` inside the env without any pip install of ``isaacsim``.

    Skipped when the ``_isaac_sim`` symlink is absent from the repo root.
    """

    @classmethod
    def setup_class(cls):
        if not shutil.which("conda"):
            pytest.skip("conda is not available")
        if not _isaacsim_source_available():
            pytest.skip("_isaac_sim symlink not found; skip source-build Isaac Sim test")

    @pytest.mark.isaacsim_source
    @pytest.mark.conda
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(1800)
    def test_install_and_train_physx(self, isaaclab_root):
        """``isaaclab.sh -c`` + ``isaaclab.sh -i physx,…`` (no isaacsim token) then train PhysX."""
        cli = str(isaaclab_root / "isaaclab.sh")
        try:
            self.create_conda_env(isaaclab_root)

            result = self.run_in_conda_env([cli, "-i", _ISAACSIM_SOURCE_INSTALL], cwd=isaaclab_root)
            _assert_no_crash(result, "isaaclab -i (physx source, conda)")

            result = self.run_in_conda_env([cli, "-p"] + _ISAACSIM_TRAIN_CMD, cwd=isaaclab_root)
            _assert_no_crash(result, "train.py physx (source, conda)")
        finally:
            self.destroy_conda_env()
