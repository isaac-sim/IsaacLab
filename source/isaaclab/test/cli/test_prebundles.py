# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Container prebundles use packages from the uv environment."""

import subprocess
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from isaaclab.cli.prebundles import repoint_prebundle_packages

pytestmark = pytest.mark.unit


def _cp(returncode: int = 0, stdout: str = "") -> mock.MagicMock:
    """Return a mock CompletedProcess with the given returncode and stdout."""
    r = mock.MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    return r


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


class TestRePointPrebundlePackages:
    """Prebundle replacement, namespace preservation, and filesystem failure behavior."""

    @pytest.fixture(autouse=True)
    def _isolate_home(self, tmp_path, monkeypatch):
        """Keep prebundle discovery away from the host's real ``~/.local/share/ov`` cache."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))

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
        """Context manager that mocks all external calls in repoint_prebundle_packages."""
        with (
            mock.patch("isaaclab.cli.prebundles.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.prebundles.extract_python_exe", return_value=python_exe),
            mock.patch("isaaclab.cli.prebundles.is_windows", return_value=False),
            mock.patch(
                "isaaclab.cli.prebundles.run_command",
                return_value=_cp(0, str(site_packages)),
            ),
        ):
            yield

    def test_no_op_when_isaac_sim_absent(self, tmp_path):
        """When Isaac Sim is not found, repoint_prebundle_packages returns immediately without touching anything."""
        with (
            mock.patch("isaaclab.cli.prebundles.extract_isaacsim_path", return_value=None),
            mock.patch("isaaclab.cli.prebundles.run_command") as mock_run,
        ):
            repoint_prebundle_packages()
        mock_run.assert_not_called()

    def test_no_op_when_no_pip_prebundle_dirs(self, tmp_path):
        """When Isaac Sim has no pip_prebundle directories, nothing is repointed."""
        isaacsim_path = tmp_path / "isaac_sim"
        isaacsim_path.mkdir()
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        py = str(tmp_path / "python")

        with self._patch(isaacsim_path, site_pkgs, py):
            repoint_prebundle_packages()

        assert list(isaacsim_path.rglob("*")) == []
        assert (site_pkgs / "torch").is_dir() and not (site_pkgs / "torch").is_symlink()

    def test_local_build_symlinks_torch_to_venv_site_packages(self, tmp_path):
        """Local _isaac_sim symlink + uv/pip venv: prebundle torch → venv site-packages/torch."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch"])
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        py = str(tmp_path / "env" / "bin" / "python")

        with self._patch(isaacsim_path, site_pkgs, py):
            repoint_prebundle_packages()

        symlink = prebundle / "torch"
        assert symlink.is_symlink(), "torch should be a symlink after repoint"
        assert symlink.resolve() == (site_pkgs / "torch").resolve()
        assert not (prebundle / "torch.bak").exists(), "repoint replaces in place — no .bak (env copy is the target)"

    def test_local_build_skips_nvidia_when_cudnn_absent_kit_python(self, tmp_path):
        """Preserve CUDA libraries when the target namespace contains only nvidia.srl."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["nvidia"])
        site_pkgs = _make_site_packages(tmp_path / "kit" / "python" / "site-packages", ["nvidia"])
        (site_pkgs / "nvidia" / "srl").mkdir()
        py = str(tmp_path / "isaac_sim" / "python.sh")

        with self._patch(isaacsim_path, site_pkgs, py):
            repoint_prebundle_packages()

        assert not (prebundle / "nvidia").is_symlink(), "nvidia must NOT be repointed when cudnn is missing"
        assert (prebundle / "nvidia").is_dir(), "Original nvidia directory must be preserved"

    def test_local_build_repoints_nvidia_when_cudnn_present_venv(self, tmp_path):
        """Repoint the nvidia namespace when the environment provides CUDA libraries."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["nvidia"])
        site_pkgs = _make_site_packages(
            tmp_path / "env",
            ["nvidia"],
            subdirs={"nvidia": ["cudnn", "cublas"]},
        )
        py = str(tmp_path / "env" / "bin" / "python")

        with self._patch(isaacsim_path, site_pkgs, py):
            repoint_prebundle_packages()

        symlink = prebundle / "nvidia"
        assert symlink.is_symlink(), "nvidia should be repointed when cudnn is present"
        assert symlink.resolve() == (site_pkgs / "nvidia").resolve()

    def test_idempotent_when_symlink_already_correct(self, tmp_path):
        """Calling repoint_prebundle_packages twice does not break the symlinks."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", [])
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        py = str(tmp_path / "env" / "bin" / "python")

        (prebundle / "torch").symlink_to(site_pkgs / "torch")
        original_target = (prebundle / "torch").resolve()

        with self._patch(isaacsim_path, site_pkgs, py):
            repoint_prebundle_packages()

        assert (prebundle / "torch").resolve() == original_target, "Correct symlink must not be changed"

    def test_updates_stale_symlink_pointing_to_old_env(self, tmp_path):
        """A symlink from a previous venv that no longer matches current site-packages is updated."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", [])
        site_pkgs = _make_site_packages(tmp_path / "env_new", ["torch"])
        old_env = _make_site_packages(tmp_path / "env_old", ["torch"])
        py = str(tmp_path / "env_new" / "bin" / "python")

        (prebundle / "torch").symlink_to(old_env / "torch")

        with self._patch(isaacsim_path, site_pkgs, py):
            repoint_prebundle_packages()

        assert (prebundle / "torch").resolve() == (site_pkgs / "torch").resolve(), "Stale symlink must be updated"

    def test_raises_when_prebundled_torch_not_neutralized(self, tmp_path):
        """Reject a surviving prebundled torch that would shadow the environment on Kit launches."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch"])
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        py = str(tmp_path / "env" / "bin" / "python")

        # Simulate the removal not taking effect (e.g. an unhandled filesystem quirk): the
        # prebundled torch stays a real directory rather than becoming a symlink.
        with self._patch(isaacsim_path, site_pkgs, py):
            with mock.patch("isaaclab.cli.prebundles._force_remove"):
                with pytest.raises(RuntimeError, match="neutralize"):
                    repoint_prebundle_packages()

    def test_repoints_across_multiple_prebundle_dirs(self, tmp_path):
        """When Isaac Sim has multiple pip_prebundle directories, each is processed."""
        isaacsim_path = tmp_path / "isaac_sim"
        isaacsim_path.mkdir()

        pb1 = isaacsim_path / "exts" / "ext_a" / "pip_prebundle"
        pb2 = isaacsim_path / "exts" / "ext_b" / "pip_prebundle"
        for pb in (pb1, pb2):
            pb.mkdir(parents=True)
            (pb / "torch").mkdir()

        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        py = str(tmp_path / "env" / "bin" / "python")

        with self._patch(isaacsim_path, site_pkgs, py):
            repoint_prebundle_packages()

        for pb in (pb1, pb2):
            assert (pb / "torch").is_symlink(), f"torch in {pb} should be repointed"

    def test_repoints_package_inside_expanded_extra_bundle(self, tmp_path):
        """Expanded extras bundles must not retain file links into a replaced package."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["newton"])
        shared_init = prebundle / "newton" / "legacy" / "__init__.py"
        shared_init.parent.mkdir()
        shared_init.write_text("")
        bundled_newton = prebundle / "newton[sim]" / "newton-wheel" / "newton"
        bundled_init = bundled_newton / "legacy" / "__init__.py"
        bundled_init.parent.mkdir(parents=True)
        bundled_init.symlink_to(shared_init)
        site_pkgs = _make_site_packages(tmp_path / "env", ["newton"])

        with self._patch(isaacsim_path, site_pkgs, str(tmp_path / "env" / "bin" / "python")):
            repoint_prebundle_packages()

        assert (prebundle / "newton").resolve() == (site_pkgs / "newton").resolve()
        assert bundled_newton.is_symlink()
        assert bundled_newton.resolve() == (site_pkgs / "newton").resolve()

    def test_copies_package_on_windows_instead_of_symlinking(self, tmp_path):
        """Copy packages on Windows without requiring symlink privileges."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch"])
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch"])
        (site_pkgs / "torch" / "version.py").write_text("__version__ = '2.10.0'")
        py = str(tmp_path / "env" / "bin" / "python")

        with (
            mock.patch("isaaclab.cli.prebundles.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.prebundles.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.prebundles.is_windows", return_value=True),
            mock.patch("isaaclab.cli.prebundles.run_command", return_value=_cp(0, str(site_pkgs))),
        ):
            repoint_prebundle_packages()

        torch_in_prebundle = prebundle / "torch"
        assert torch_in_prebundle.is_dir(), "torch should be a directory (copy) on Windows"
        assert not torch_in_prebundle.is_symlink(), "torch must not be a symlink on Windows"
        assert (torch_in_prebundle / "version.py").exists(), "Copied file should be present"

    def test_oserror_on_one_package_does_not_abort_others(self, tmp_path):
        """An OSError while repointing one package is logged and processing continues for others."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch", "torchvision"])
        site_pkgs = _make_site_packages(tmp_path / "env", ["torch", "torchvision"])
        py = str(tmp_path / "env" / "bin" / "python")

        original_symlink_to = Path.symlink_to
        call_count: list[int] = [0]

        def _selective_symlink(self_path: Path, target: Path, **kwargs) -> None:
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("Permission denied")
            return original_symlink_to(self_path, target, **kwargs)

        with (
            mock.patch("isaaclab.cli.prebundles.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.prebundles.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.prebundles.is_windows", return_value=False),
            mock.patch("isaaclab.cli.prebundles.run_command", return_value=_cp(0, str(site_pkgs))),
            mock.patch.object(Path, "symlink_to", _selective_symlink),
        ):
            repoint_prebundle_packages()

        assert (prebundle / "torchvision").is_symlink(), "torchvision must succeed after torch OSError"

    def test_skips_gracefully_when_site_packages_probe_fails(self, tmp_path):
        """When the site-packages probe subprocess fails, repoint_prebundle_packages is a no-op."""
        isaacsim_path, prebundle = self._sim_with_prebundle(tmp_path / "sim", ["torch"])
        py = str(tmp_path / "python")

        with (
            mock.patch("isaaclab.cli.prebundles.extract_isaacsim_path", return_value=isaacsim_path),
            mock.patch("isaaclab.cli.prebundles.extract_python_exe", return_value=py),
            mock.patch("isaaclab.cli.prebundles.is_windows", return_value=False),
            mock.patch("isaaclab.cli.prebundles.run_command", return_value=_cp(returncode=1, stdout="")),
        ):
            repoint_prebundle_packages()

        assert not (prebundle / "torch").is_symlink(), "No symlink should be created when probe fails"
