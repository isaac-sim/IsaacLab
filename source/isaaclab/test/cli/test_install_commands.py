# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for install command functions.

Covers all combinations of:
- Python environment types: uv venv, pip venv, conda, Isaac Sim kit Python, system Python
- Isaac Sim installation methods: local _isaac_sim symlink, pip-installed isaacsim, none
"""

import subprocess
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from isaaclab.cli.commands.install import (
    _PREBUNDLE_REPOINT_PACKAGES,
    _ensure_cuda_torch,
    _maybe_uninstall_prebundled_torch,
    _repoint_prebundle_packages,
    _torch_first_on_sys_path_is_prebundle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cp(returncode: int = 0, stdout: str = "") -> mock.MagicMock:
    """Return a mock CompletedProcess with the given returncode and stdout."""
    r = mock.MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    return r


def _make_prebundle(base: Path, packages: list[str]) -> Path:
    """Create a fake pip_prebundle directory populated with the given package dirs."""
    prebundle = base / "pip_prebundle"
    prebundle.mkdir(parents=True)
    for pkg in packages:
        (prebundle / pkg).mkdir()
    return prebundle


def _make_site_packages(
    base: Path,
    packages: list[str],
    subdirs: dict[str, list[str]] | None = None,
) -> Path:
    """Create a fake site-packages directory.

    Args:
        packages: Top-level package directory names to create.
        subdirs: Optional mapping of package name → list of subdirectory names to create inside it.
    """
    site_pkgs = base / "site-packages"
    site_pkgs.mkdir(parents=True, exist_ok=True)
    for pkg in packages:
        (site_pkgs / pkg).mkdir(exist_ok=True)
    for pkg, subs in (subdirs or {}).items():
        for sub in subs:
            (site_pkgs / pkg / sub).mkdir(parents=True, exist_ok=True)
    return site_pkgs


# ---------------------------------------------------------------------------
# _torch_first_on_sys_path_is_prebundle
# ---------------------------------------------------------------------------


class TestTorchProbe:
    """Tests for :func:`_torch_first_on_sys_path_is_prebundle`.

    The function shells out to ``python_exe -c <probe>`` and interprets the
    subprocess exit code: 1 → prebundle is first; 0 → it is not.
    """

    def test_returns_true_when_prebundle_first(self, tmp_path):
        """Probe exits 1 → the first torch on sys.path is under a pip_prebundle directory."""
        with mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(returncode=1)):
            result = _torch_first_on_sys_path_is_prebundle(
                str(tmp_path / "python"),
                env={"PYTHONPATH": "/fake/extsDeprecated/pip_prebundle"},
            )
        assert result is True

    def test_returns_false_when_site_packages_first(self, tmp_path):
        """Probe exits 0 → the first torch on sys.path is in regular site-packages."""
        with mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(returncode=0)):
            result = _torch_first_on_sys_path_is_prebundle(
                str(tmp_path / "python"),
                env={"PYTHONPATH": "/conda/lib/python3.12/site-packages"},
            )
        assert result is False

    def test_returns_false_when_torch_not_found_anywhere(self, tmp_path):
        """Probe exits 0 (no torch on sys.path at all) → returns False."""
        with mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(returncode=0)):
            result = _torch_first_on_sys_path_is_prebundle(
                str(tmp_path / "python"),
                env={},
            )
        assert result is False

    def test_passes_env_to_subprocess(self, tmp_path):
        """The custom env dict is forwarded to run_command."""
        env_sent = {"PYTHONPATH": "/some/path"}
        with mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(0)) as mock_run:
            _torch_first_on_sys_path_is_prebundle(str(tmp_path / "python"), env=env_sent)
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("env") == env_sent or (
            len(call_kwargs.args) > 1 and call_kwargs.args[1] == env_sent
        )


# ---------------------------------------------------------------------------
# _maybe_uninstall_prebundled_torch
# ---------------------------------------------------------------------------


class TestMaybeUninstallTorch:
    """Tests for :func:`_maybe_uninstall_prebundled_torch`."""

    def test_does_not_uninstall_when_probe_false(self, tmp_path):
        """When the probe returns False, no pip uninstall command is issued."""
        py = str(tmp_path / "python")
        with (
            mock.patch(
                "isaaclab.cli.commands.install._torch_first_on_sys_path_is_prebundle",
                return_value=False,
            ),
            mock.patch("isaaclab.cli.commands.install.run_command") as mock_run,
        ):
            _maybe_uninstall_prebundled_torch(py, [py, "-m", "pip"], using_uv=False, probe_env={})
        mock_run.assert_not_called()

    def test_uninstalls_torch_stack_with_minus_y_for_pip(self, tmp_path):
        """When probe returns True and pip is in use, uninstall includes -y flag."""
        py = str(tmp_path / "python")
        with (
            mock.patch(
                "isaaclab.cli.commands.install._torch_first_on_sys_path_is_prebundle",
                return_value=True,
            ),
            mock.patch("isaaclab.cli.commands.install.run_command") as mock_run,
        ):
            _maybe_uninstall_prebundled_torch(py, [py, "-m", "pip"], using_uv=False, probe_env={})
        mock_run.assert_called_once()
        issued = mock_run.call_args[0][0]
        assert "uninstall" in issued
        assert "-y" in issued
        assert "torch" in issued
        assert "torchvision" in issued
        assert "torchaudio" in issued

    def test_uninstalls_torch_stack_without_minus_y_for_uv(self, tmp_path):
        """When probe returns True and uv pip is in use, uninstall omits -y (uv doesn't accept it)."""
        with (
            mock.patch(
                "isaaclab.cli.commands.install._torch_first_on_sys_path_is_prebundle",
                return_value=True,
            ),
            mock.patch("isaaclab.cli.commands.install.run_command") as mock_run,
        ):
            _maybe_uninstall_prebundled_torch("/fake/python", ["uv", "pip"], using_uv=True, probe_env={})
        issued = mock_run.call_args[0][0]
        assert "uninstall" in issued
        assert "-y" not in issued

    def test_probe_receives_original_pythonpath(self, tmp_path):
        """The probe_env dict is forwarded unchanged to the torch-probe function."""
        py = str(tmp_path / "python")
        probe_env = {"PYTHONPATH": "/a/extsDeprecated/pip_prebundle:/b/site-packages"}
        with mock.patch(
            "isaaclab.cli.commands.install._torch_first_on_sys_path_is_prebundle",
            return_value=False,
        ) as mock_probe:
            _maybe_uninstall_prebundled_torch(py, [py, "-m", "pip"], using_uv=False, probe_env=probe_env)
        mock_probe.assert_called_once_with(py, env=probe_env)


# ---------------------------------------------------------------------------
# _ensure_cuda_torch  — architecture × environment combinations
# ---------------------------------------------------------------------------


class TestEnsureCudaTorch:
    """Tests for :func:`_ensure_cuda_torch` across architectures and environment types.

    Combinations tested:
    - Architecture:  x86 (cu128) vs ARM (cu130)
    - Pip command:   ``python -m pip`` (venv/conda/kit) vs ``uv pip`` (uv venv)
    - Torch state:   already installed at correct version; wrong CUDA tag; not installed
    """

    # ---- x86 scenarios -------------------------------------------------------

    def test_x86_skips_install_when_correct_version_present(self, tmp_path):
        """x86: torch 2.10.0+cu128 already installed → pip install is not called."""
        py = str(tmp_path / "python")
        pip_cmd = [py, "-m", "pip"]
        pip_show_out = "Name: torch\nVersion: 2.10.0+cu128\n"

        with (
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.get_pip_command", return_value=pip_cmd),
            mock.patch("isaaclab.cli.commands.install.is_arm", return_value=False),
            mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(0, pip_show_out)) as mock_run,
        ):
            _ensure_cuda_torch()

        # Only the initial ``pip show torch`` call; no install.
        assert mock_run.call_count == 1
        assert "show" in mock_run.call_args[0][0]

    def test_x86_installs_cu128_when_torch_missing(self, tmp_path):
        """x86: no torch installed → installs torch+cu128 from pytorch.org/whl/cu128."""
        py = str(tmp_path / "python")
        pip_cmd = [py, "-m", "pip"]
        calls: list[list[str]] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return _cp(0, "")  # pip show returns nothing → torch absent

        with (
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.get_pip_command", return_value=pip_cmd),
            mock.patch("isaaclab.cli.commands.install.is_arm", return_value=False),
            mock.patch("isaaclab.cli.commands.install.run_command", side_effect=_run),
        ):
            _ensure_cuda_torch()

        install_cmds = [c for c in calls if "install" in c]
        combined = " ".join(str(t) for c in install_cmds for t in c)
        assert "cu128" in combined
        assert "torch" in combined

    def test_x86_reinstalls_when_wrong_cuda_tag(self, tmp_path):
        """x86: torch+cu130 installed (ARM build) → uninstalls and reinstalls as cu128."""
        py = str(tmp_path / "python")
        pip_cmd = [py, "-m", "pip"]
        calls: list[list[str]] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            stdout = "Name: torch\nVersion: 2.10.0+cu130\n" if "show" in cmd else ""
            return _cp(0, stdout)

        with (
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.get_pip_command", return_value=pip_cmd),
            mock.patch("isaaclab.cli.commands.install.is_arm", return_value=False),
            mock.patch("isaaclab.cli.commands.install.run_command", side_effect=_run),
        ):
            _ensure_cuda_torch()

        assert any("uninstall" in c for c in calls), "Expected an uninstall call"
        install_cmds = [c for c in calls if "install" in c]
        combined = " ".join(str(t) for c in install_cmds for t in c)
        assert "cu128" in combined

    # ---- ARM scenarios -------------------------------------------------------

    def test_arm_installs_cu130_when_torch_missing(self, tmp_path):
        """ARM: no torch installed → installs torch+cu130 from pytorch.org/whl/cu130."""
        py = str(tmp_path / "python")
        pip_cmd = [py, "-m", "pip"]
        calls: list[list[str]] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return _cp(0, "")

        with (
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.get_pip_command", return_value=pip_cmd),
            mock.patch("isaaclab.cli.commands.install.is_arm", return_value=True),
            mock.patch("isaaclab.cli.commands.install.run_command", side_effect=_run),
        ):
            _ensure_cuda_torch()

        install_cmds = [c for c in calls if "install" in c]
        combined = " ".join(str(t) for c in install_cmds for t in c)
        assert "cu130" in combined

    def test_arm_skips_install_when_correct_version_present(self, tmp_path):
        """ARM: torch 2.10.0+cu130 already installed → pip install is not called."""
        py = str(tmp_path / "python")
        pip_cmd = [py, "-m", "pip"]
        pip_show_out = "Name: torch\nVersion: 2.10.0+cu130\n"

        with (
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.get_pip_command", return_value=pip_cmd),
            mock.patch("isaaclab.cli.commands.install.is_arm", return_value=True),
            mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(0, pip_show_out)) as mock_run,
        ):
            _ensure_cuda_torch()

        assert mock_run.call_count == 1

    def test_arm_reinstalls_when_wrong_cuda_tag(self, tmp_path):
        """ARM: torch+cu128 installed (x86 build) → uninstalls and reinstalls as cu130."""
        py = str(tmp_path / "python")
        pip_cmd = [py, "-m", "pip"]
        calls: list[list[str]] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            stdout = "Name: torch\nVersion: 2.10.0+cu128\n" if "show" in cmd else ""
            return _cp(0, stdout)

        with (
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.get_pip_command", return_value=pip_cmd),
            mock.patch("isaaclab.cli.commands.install.is_arm", return_value=True),
            mock.patch("isaaclab.cli.commands.install.run_command", side_effect=_run),
        ):
            _ensure_cuda_torch()

        assert any("uninstall" in c for c in calls)
        install_cmds = [c for c in calls if "install" in c]
        combined = " ".join(str(t) for c in install_cmds for t in c)
        assert "cu130" in combined

    # ---- uv venv environment ------------------------------------------------

    def test_uv_venv_uses_uv_pip_command(self, tmp_path):
        """In a uv venv get_pip_command returns ['uv', 'pip'] and uninstall omits -y."""
        py = str(tmp_path / "python")
        calls: list[list[str]] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return _cp(0, "")  # no current torch → triggers install

        with (
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.get_pip_command", return_value=["uv", "pip"]),
            mock.patch("isaaclab.cli.commands.install.is_arm", return_value=False),
            mock.patch("isaaclab.cli.commands.install.run_command", side_effect=_run),
        ):
            _ensure_cuda_torch()

        assert calls[0][0] == "uv", "Expected uv as the pip command prefix"
        uninstall_calls = [c for c in calls if "uninstall" in c]
        assert uninstall_calls, "Expected an uninstall call before reinstall"
        assert "-y" not in uninstall_calls[0], "uv pip uninstall must not include -y"

    # ---- conda / pip venv / kit Python environments -------------------------

    def test_conda_uses_python_m_pip_with_minus_y(self, tmp_path):
        """In a conda env (no uv), get_pip_command returns python -m pip; uninstall uses -y."""
        py = str(tmp_path / "conda" / "bin" / "python")
        pip_cmd = [py, "-m", "pip"]
        calls: list[list[str]] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return _cp(0, "")

        with (
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.get_pip_command", return_value=pip_cmd),
            mock.patch("isaaclab.cli.commands.install.is_arm", return_value=False),
            mock.patch("isaaclab.cli.commands.install.run_command", side_effect=_run),
        ):
            _ensure_cuda_torch()

        uninstall_calls = [c for c in calls if "uninstall" in c]
        assert uninstall_calls
        assert "-y" in uninstall_calls[0], "pip uninstall must include -y"
        assert py in uninstall_calls[0], "Expected python exe in pip command"

    def test_kit_python_uses_python_sh_as_pip_prefix(self, tmp_path):
        """With Isaac Sim's kit Python, python.sh is the executable prefix in the pip command."""
        python_sh = str(tmp_path / "_isaac_sim" / "python.sh")
        pip_cmd = [python_sh, "-m", "pip"]
        calls: list[list[str]] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return _cp(0, "")

        with (
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=python_sh),
            mock.patch("isaaclab.cli.commands.install.get_pip_command", return_value=pip_cmd),
            mock.patch("isaaclab.cli.commands.install.is_arm", return_value=False),
            mock.patch("isaaclab.cli.commands.install.run_command", side_effect=_run),
        ):
            _ensure_cuda_torch()

        assert calls[0][0] == python_sh


# ---------------------------------------------------------------------------
# _repoint_prebundle_packages  — Isaac Sim install method × venv type
# ---------------------------------------------------------------------------


class TestRePointPrebundlePackages:
    """Tests for :func:`_repoint_prebundle_packages`.

    Covers all combinations of:
    - Isaac Sim installation method: local _isaac_sim symlink, pip-installed isaacsim, none
    - Python environment / site-packages source: uv venv, pip venv, conda, kit Python
    - nvidia namespace package special handling: cudnn present vs absent
    """

    # ---- shared fixtures / helpers ------------------------------------------

    def _sim_with_prebundle(self, base: Path, packages: list[str]) -> tuple[Path, Path]:
        """Create a minimal fake Isaac Sim tree containing a pip_prebundle dir.

        Returns ``(isaacsim_path, prebundle_dir)``.
        """
        isaacsim_path = base / "isaac_sim"
        isaacsim_path.mkdir(parents=True)
        prebundle = isaacsim_path / "exts" / "some.ext" / "pip_prebundle"
        prebundle.mkdir(parents=True)
        for pkg in packages:
            (prebundle / pkg).mkdir()
        return isaacsim_path, prebundle

    @contextmanager
    def _patch(self, isaacsim_path: Path | None, site_packages: Path, python_exe: str):
        """Context manager that mocks all external calls in _repoint_prebundle_packages."""
        with (
            mock.patch("isaaclab.cli.commands.install.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=python_exe),
            mock.patch("isaaclab.cli.commands.install.is_windows", return_value=False),
            mock.patch(
                "isaaclab.cli.commands.install.run_command",
                return_value=_cp(0, str(site_packages)),
            ),
        ):
            yield

    # ---- no Isaac Sim --------------------------------------------------------

    def test_no_op_when_isaac_sim_absent(self, tmp_path):
        """When Isaac Sim is not found, _repoint_prebundle_packages returns immediately without touching anything."""
        with (
            mock.patch("isaaclab.cli.commands.install.extract_isaacsim_path", return_value=None),
            mock.patch("isaaclab.cli.commands.install.run_command") as mock_run,
        ):
            _repoint_prebundle_packages()
        mock_run.assert_not_called()

    # ---- no pip_prebundle directories ----------------------------------------

    def test_no_op_when_no_pip_prebundle_dirs(self, tmp_path):
        """When Isaac Sim has no pip_prebundle directories, nothing is repointed."""
        isaacsim_path = tmp_path / "isaac_sim"
        isaacsim_path.mkdir()
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        py = str(tmp_path / "python")

        with self._patch(isaacsim_path, site_pkgs, py):
            _repoint_prebundle_packages()

        assert not (site_pkgs.parent / "pip_prebundle" / "torch").exists()

    # ---- local _isaac_sim symlink (local build) ------------------------------

    def test_local_build_symlinks_torch_to_venv_site_packages(self, tmp_path):
        """Local _isaac_sim symlink + uv/pip venv: prebundle torch → venv site-packages/torch."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch"])
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        py = str(tmp_path / "env" / "bin" / "python")

        with self._patch(isaacsim_path, site_pkgs, py):
            _repoint_prebundle_packages()

        symlink = prebundle / "torch"
        assert symlink.is_symlink(), "torch should be a symlink after repoint"
        assert symlink.resolve() == (site_pkgs / "torch").resolve()
        assert (prebundle / "torch.bak").is_dir(), "Original torch should be backed up"

    def test_local_build_skips_nvidia_when_cudnn_absent_kit_python(self, tmp_path):
        """Local build + kit Python: site-packages/nvidia has only 'srl' (no cudnn) → nvidia NOT repointed.

        This is the real-world failure mode that caused the libcudnn.so.9 import error:
        kit Python's site-packages/nvidia has only the 'srl' namespace sub-package, so
        replacing the prebundle's nvidia/ (which contains the CUDA shared libraries) with
        a symlink to that stripped-down directory would make libcudnn.so.9 unreachable.
        """
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["nvidia"])
        # Simulate kit Python's site-packages: nvidia/ exists but contains only 'srl'
        site_pkgs = _make_site_packages(tmp_path / "kit" / "python" / "site-packages", ["nvidia"])
        (site_pkgs / "nvidia" / "srl").mkdir()
        py = str(tmp_path / "isaac_sim" / "python.sh")

        with self._patch(isaacsim_path, site_pkgs, py):
            _repoint_prebundle_packages()

        assert not (prebundle / "nvidia").is_symlink(), "nvidia must NOT be repointed when cudnn is missing"
        assert (prebundle / "nvidia").is_dir(), "Original nvidia directory must be preserved"

    def test_local_build_repoints_nvidia_when_cudnn_present_venv(self, tmp_path):
        """Local build + CUDA-capable venv: site-packages/nvidia has cudnn → nvidia IS repointed.

        This covers the conda or pip venv case where the user installed torch+cu128/cu130
        with its nvidia-cudnn-cu12 dependency, giving site-packages/nvidia/cudnn/.
        """
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["nvidia"])
        # Full CUDA venv: nvidia/ has cudnn and cublas
        site_pkgs = _make_site_packages(
            tmp_path / "env",
            ["nvidia"],
            subdirs={"nvidia": ["cudnn", "cublas"]},
        )
        py = str(tmp_path / "env" / "bin" / "python")

        with self._patch(isaacsim_path, site_pkgs, py):
            _repoint_prebundle_packages()

        symlink = prebundle / "nvidia"
        assert symlink.is_symlink(), "nvidia should be repointed when cudnn is present"
        assert symlink.resolve() == (site_pkgs / "nvidia").resolve()

    def test_idempotent_when_symlink_already_correct(self, tmp_path):
        """Calling _repoint_prebundle_packages twice does not break the symlinks."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", [])
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        py = str(tmp_path / "env" / "bin" / "python")

        # Pre-create the correct symlink (as if a previous install already ran).
        (prebundle / "torch").symlink_to(site_pkgs / "torch")
        original_target = (prebundle / "torch").resolve()

        with self._patch(isaacsim_path, site_pkgs, py):
            _repoint_prebundle_packages()

        assert (prebundle / "torch").resolve() == original_target, "Correct symlink must not be changed"

    def test_updates_stale_symlink_pointing_to_old_env(self, tmp_path):
        """A symlink from a previous venv that no longer matches current site-packages is updated."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", [])
        site_pkgs = _make_site_packages(tmp_path / "env_new", ["torch"])
        old_env = _make_site_packages(tmp_path / "env_old", ["torch"])
        py = str(tmp_path / "env_new" / "bin" / "python")

        # Pre-create a stale symlink pointing at the old env.
        (prebundle / "torch").symlink_to(old_env / "torch")

        with self._patch(isaacsim_path, site_pkgs, py):
            _repoint_prebundle_packages()

        assert (prebundle / "torch").resolve() == (site_pkgs / "torch").resolve(), "Stale symlink must be updated"

    def test_removes_old_backup_before_renaming(self, tmp_path):
        """A pre-existing .bak directory is removed before the current package is backed up."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch"])
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        py = str(tmp_path / "env" / "bin" / "python")

        # Simulate leftover backup from a previous partial install.
        old_backup = prebundle / "torch.bak"
        old_backup.mkdir()
        (old_backup / "stale_file.py").touch()

        with self._patch(isaacsim_path, site_pkgs, py):
            _repoint_prebundle_packages()

        assert (prebundle / "torch").is_symlink(), "torch must be repointed"
        # The old backup was replaced by the fresh backup.
        assert (prebundle / "torch.bak").is_dir()

    # ---- pip-installed isaacsim (path found via import probe) ----------------

    def test_pip_isaacsim_symlinks_torch(self, tmp_path):
        """pip-installed isaacsim: extract_isaacsim_path() returns its directory and torch is repointed."""
        # With pip-installed isaacsim the path may be inside site-packages rather than a symlink
        # at the repo root, but _repoint_prebundle_packages treats it identically.
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "pip_isaacsim", ["torch"])
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        py = str(tmp_path / "env" / "bin" / "python")

        with self._patch(isaacsim_path, site_pkgs, py):
            _repoint_prebundle_packages()

        assert (prebundle / "torch").is_symlink()
        assert (prebundle / "torch").resolve() == (site_pkgs / "torch").resolve()

    def test_pip_isaacsim_skips_nvidia_without_cudnn(self, tmp_path):
        """pip-installed isaacsim + no cudnn in site-packages → nvidia prebundle preserved."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "pip_isaacsim", ["nvidia"])
        # site-packages has nvidia/ but without a cudnn sub-package
        site_pkgs = _make_site_packages(tmp_path / "env", ["nvidia"])
        py = str(tmp_path / "env" / "bin" / "python")

        with self._patch(isaacsim_path, site_pkgs, py):
            _repoint_prebundle_packages()

        assert not (prebundle / "nvidia").is_symlink(), "nvidia must not be repointed without cudnn"

    # ---- different venv types ------------------------------------------------

    def test_uv_venv_repoints_torch_using_venv_site_packages(self, tmp_path):
        """uv venv: site-packages inside VIRTUAL_ENV is used as the symlink target."""
        venv_site = tmp_path / "env_uv" / "lib" / "python3.12" / "site-packages"
        venv_site.mkdir(parents=True)
        (venv_site / "torch").mkdir()
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch"])
        py = str(tmp_path / "env_uv" / "bin" / "python")

        with (
            mock.patch("isaaclab.cli.commands.install.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.is_windows", return_value=False),
            mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(0, str(venv_site))),
        ):
            _repoint_prebundle_packages()

        assert (prebundle / "torch").is_symlink()
        assert (prebundle / "torch").resolve() == (venv_site / "torch").resolve()

    def test_conda_repoints_torch_using_conda_site_packages(self, tmp_path):
        """conda env: site-packages inside CONDA_PREFIX is used as the symlink target."""
        conda_site = tmp_path / "conda" / "lib" / "python3.12" / "site-packages"
        conda_site.mkdir(parents=True)
        (conda_site / "torch").mkdir()
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch"])
        py = str(tmp_path / "conda" / "bin" / "python")

        with (
            mock.patch("isaaclab.cli.commands.install.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.is_windows", return_value=False),
            mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(0, str(conda_site))),
        ):
            _repoint_prebundle_packages()

        assert (prebundle / "torch").is_symlink()
        assert (prebundle / "torch").resolve() == (conda_site / "torch").resolve()

    def test_conda_repoints_nvidia_when_full_cuda_torch_installed(self, tmp_path):
        """conda env with nvidia-cudnn-cu12 installed: nvidia/ is repointed because cudnn subdir exists."""
        conda_site = tmp_path / "conda" / "lib" / "python3.12" / "site-packages"
        conda_site.mkdir(parents=True)
        (conda_site / "nvidia").mkdir()
        (conda_site / "nvidia" / "cudnn").mkdir()
        (conda_site / "nvidia" / "cublas").mkdir()
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["nvidia"])
        py = str(tmp_path / "conda" / "bin" / "python")

        with (
            mock.patch("isaaclab.cli.commands.install.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.is_windows", return_value=False),
            mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(0, str(conda_site))),
        ):
            _repoint_prebundle_packages()

        assert (prebundle / "nvidia").is_symlink()

    def test_conda_skips_nvidia_when_no_cudnn(self, tmp_path):
        """conda env without CUDA torch: site-packages/nvidia lacks cudnn → nvidia not repointed."""
        conda_site = tmp_path / "conda" / "lib" / "python3.12" / "site-packages"
        conda_site.mkdir(parents=True)
        (conda_site / "nvidia").mkdir()  # exists but no cudnn inside
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["nvidia"])
        py = str(tmp_path / "conda" / "bin" / "python")

        with (
            mock.patch("isaaclab.cli.commands.install.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.is_windows", return_value=False),
            mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(0, str(conda_site))),
        ):
            _repoint_prebundle_packages()

        assert not (prebundle / "nvidia").is_symlink()

    # ---- multiple prebundle directories -------------------------------------

    def test_repoints_across_multiple_prebundle_dirs(self, tmp_path):
        """When Isaac Sim has multiple pip_prebundle directories, each is processed."""
        isaacsim_path = tmp_path / "isaac_sim"
        isaacsim_path.mkdir()

        # Two separate extension pip_prebundle dirs, each with torch.
        pb1 = isaacsim_path / "exts" / "ext_a" / "pip_prebundle"
        pb2 = isaacsim_path / "exts" / "ext_b" / "pip_prebundle"
        for pb in (pb1, pb2):
            pb.mkdir(parents=True)
            (pb / "torch").mkdir()

        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        py = str(tmp_path / "env" / "bin" / "python")

        with self._patch(isaacsim_path, site_pkgs, py):
            _repoint_prebundle_packages()

        for pb in (pb1, pb2):
            assert (pb / "torch").is_symlink(), f"torch in {pb} should be repointed"

    # ---- Windows: copy instead of symlink -----------------------------------

    def test_copies_package_on_windows_instead_of_symlinking(self, tmp_path):
        """On Windows, packages are copied rather than symlinked (Windows doesn't support posix symlinks)."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch"])
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        (site_pkgs / "torch" / "version.py").write_text("__version__ = '2.10.0'")
        py = str(tmp_path / "env" / "bin" / "python")

        with (
            mock.patch("isaaclab.cli.commands.install.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.is_windows", return_value=True),
            mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(0, str(site_pkgs))),
        ):
            _repoint_prebundle_packages()

        torch_in_prebundle = prebundle / "torch"
        assert torch_in_prebundle.is_dir(), "torch should be a directory (copy) on Windows"
        assert not torch_in_prebundle.is_symlink(), "torch must not be a symlink on Windows"
        assert (torch_in_prebundle / "version.py").exists(), "Copied file should be present"

    # ---- error handling -----------------------------------------------------

    def test_oserror_on_one_package_does_not_abort_others(self, tmp_path):
        """An OSError while repointing one package is logged and processing continues for others."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch", "torchvision"])
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch", "torchvision"])
        py = str(tmp_path / "env" / "bin" / "python")

        original_symlink_to = Path.symlink_to
        call_count: list[int] = [0]

        def _selective_symlink(self_path: Path, target: Path, **kwargs) -> None:
            call_count[0] += 1
            # Fail on the first symlink_to call (torch) but succeed for others.
            if call_count[0] == 1:
                raise OSError("Permission denied")
            return original_symlink_to(self_path, target, **kwargs)

        with (
            mock.patch("isaaclab.cli.commands.install.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.is_windows", return_value=False),
            mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(0, str(site_pkgs))),
            mock.patch.object(Path, "symlink_to", _selective_symlink),
        ):
            _repoint_prebundle_packages()  # must not raise

        # torchvision (second package) must still be repointed despite torch failure.
        assert (prebundle / "torchvision").is_symlink(), "torchvision must succeed after torch OSError"

    def test_skips_gracefully_when_site_packages_probe_fails(self, tmp_path):
        """When the site-packages probe subprocess fails, _repoint_prebundle_packages is a no-op."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch"])
        py = str(tmp_path / "python")

        with (
            mock.patch("isaaclab.cli.commands.install.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.commands.install.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.commands.install.is_windows", return_value=False),
            # Probe subprocess exits non-zero
            mock.patch("isaaclab.cli.commands.install.run_command", return_value=_cp(returncode=1, stdout="")),
        ):
            _repoint_prebundle_packages()

        assert not (prebundle / "torch").is_symlink(), "No symlink should be created when probe fails"

    # ---- all packages in the repoint list are covered -----------------------

    @pytest.mark.parametrize("pkg_name", [p for p in _PREBUNDLE_REPOINT_PACKAGES if p != "nvidia"])
    def test_all_non_nvidia_packages_are_repointed(self, tmp_path, pkg_name):
        """Every non-nvidia entry in _PREBUNDLE_REPOINT_PACKAGES is repointed when it exists."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", [pkg_name])
        site_pkgs = _make_site_packages(tmp_path / "env", [pkg_name])
        py = str(tmp_path / "env" / "bin" / "python")

        with self._patch(isaacsim_path, site_pkgs, py):
            _repoint_prebundle_packages()

        assert (prebundle / pkg_name).is_symlink(), f"{pkg_name} should be repointed"
        assert (prebundle / pkg_name).resolve() == (site_pkgs / pkg_name).resolve()
