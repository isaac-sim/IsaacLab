# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ``nvidia-smi topo -m`` parser behind :func:`gpu_pairs_by_topology`.

Fixtures are verbatim output from three differently-shaped hosts, so the parser
is covered on any machine -- including single-GPU CI, where the multi-GPU smoke
tests that consume it cannot run.
"""

from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from isaaclab.test.utils import gpu_pairs_by_topology

# 4x L40, one GPU alone on socket 0. Carries a NIC column whose cells reuse the
# same link-class tokens -- reading past the GPU columns invents a pair against
# a GPU that does not exist.
_TOPO_1V3_WITH_NIC = """\t\tGPU0\tGPU1\tGPU2\tGPU3\tNIC0\tCPU Affinity\tNUMA Affinity
GPU0\t X \tSYS\tSYS\tSYS\tNODE\t0-31,64-95\t0
GPU1\tSYS\t X \tPIX\tPIX\tSYS\t32-63,96-127\t1
GPU2\tSYS\tPIX\t X \tPIX\tSYS\t32-63,96-127\t1
GPU3\tSYS\tPIX\tPIX\t X \tSYS\t32-63,96-127\t1
NIC0\tNODE\tSYS\tSYS\tSYS\t X
"""

# 8x L40 split 4v4 across two sockets.
_TOPO_4V4 = """\t\tGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\tCPU Affinity
GPU0\t X \tPIX\tPIX\tPIX\tSYS\tSYS\tSYS\tSYS\t0-27
GPU1\tPIX\t X \tPIX\tPIX\tSYS\tSYS\tSYS\tSYS\t0-27
GPU2\tPIX\tPIX\t X \tPIX\tSYS\tSYS\tSYS\tSYS\t0-27
GPU3\tPIX\tPIX\tPIX\t X \tSYS\tSYS\tSYS\tSYS\t0-27
GPU4\tSYS\tSYS\tSYS\tSYS\t X \tPIX\tPIX\tPIX\t28-55
GPU5\tSYS\tSYS\tSYS\tSYS\tPIX\t X \tPIX\tPIX\t28-55
GPU6\tSYS\tSYS\tSYS\tSYS\tPIX\tPIX\t X \tPIX\t28-55
GPU7\tSYS\tSYS\tSYS\tSYS\tPIX\tPIX\tPIX\t X \t28-55
"""

# 4x RTX 6000 Ada, single socket: every pair is PHB or NODE, neither of which is
# measured for the defect the classification gates.
_TOPO_SINGLE_SOCKET = """\t\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity
GPU0\t X \tPHB\tNODE\tNODE\t0-47\t0
GPU1\tPHB\t X \tNODE\tNODE\t0-47\t0
GPU2\tNODE\tNODE\t X \tNODE\t0-47\t0
GPU3\tNODE\tNODE\tNODE\t X \t0-47\t0
"""

_TOPO_NVLINK = """\t\tGPU0\tGPU1\tCPU Affinity
GPU0\t X \tNV18\t0-95
GPU1\tNV18\t X \t0-95
"""


# A row naming a GPU the header never lists: the matrix is partial or inconsistent.
_TOPO_ROW_OUTSIDE_HEADER = """\t\tGPU0\tGPU1\tCPU Affinity
GPU0\t X \tSYS\t0-95
GPU9\tSYS\t X \t0-95
"""

# Header advertises four GPUs but only two rows are present.
_TOPO_TRUNCATED = """\t\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity
GPU0\t X \tPIX\tPIX\tPIX\t0-27
GPU1\tPIX\t X \tPIX\tPIX\t0-27
"""

_NVIDIA_SMI_L_DISCRETE = "GPU 0: NVIDIA L40 (UUID: GPU-aaa)\nGPU 1: NVIDIA L40 (UUID: GPU-bbb)\n"
_NVIDIA_SMI_L_MIG = (
    "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-aaa)\n"
    "  MIG 3g.20gb     Device  0: (UUID: MIG-bbb)\n"
    "  MIG 3g.20gb     Device  1: (UUID: MIG-ccc)\n"
)


def _with_topo(stdout: str, returncode: int = 0, devices: str = _NVIDIA_SMI_L_DISCRETE):
    """Patch ``subprocess.run`` so the parser sees ``stdout`` as nvidia-smi output.

    ``devices`` stands in for ``nvidia-smi -L``, which the parser consults first to
    rule out a MIG host.
    """

    def _run(cmd, *args, **kwargs):
        out = devices if "-L" in cmd else stdout
        code = 0 if "-L" in cmd else returncode
        return subprocess.CompletedProcess(args=cmd, returncode=code, stdout=out, stderr="")

    return mock.patch("isaaclab.test.utils.devices.subprocess.run", side_effect=_run)


class TestGpuPairsByTopology:
    """Classification of GPU pairs from nvidia-smi topology output."""

    def test_two_socket_split_reports_both_classes(self) -> None:
        with _with_topo(_TOPO_4V4):
            assert gpu_pairs_by_topology() == {"SAME_SWITCH": (0, 1), "CROSS_SOCKET": (0, 4)}

    def test_nic_column_does_not_invent_a_gpu_pair(self) -> None:
        """NIC cells carry the same tokens as GPU cells and must not be classified."""
        with _with_topo(_TOPO_1V3_WITH_NIC):
            pairs = gpu_pairs_by_topology()
        assert pairs == {"CROSS_SOCKET": (0, 1), "SAME_SWITCH": (1, 2)}
        # The host has 4 GPUs; a pair naming index 4 would come from the NIC column.
        assert all(max(pair) < 4 for pair in pairs.values())

    def test_single_socket_reports_only_unknown(self) -> None:
        """PHB/NODE are unmeasured for this defect, so neither camera case may run."""
        with _with_topo(_TOPO_SINGLE_SOCKET):
            pairs = gpu_pairs_by_topology()
        assert pairs == {"UNKNOWN": (0, 1)}
        assert "CROSS_SOCKET" not in pairs
        assert "SAME_SWITCH" not in pairs

    def test_nvlink_counts_as_same_switch(self) -> None:
        with _with_topo(_TOPO_NVLINK):
            assert gpu_pairs_by_topology() == {"SAME_SWITCH": (0, 1)}

    @pytest.mark.parametrize(
        "stdout,returncode",
        [("", 0), ("no topology here", 0), (_TOPO_4V4, 1)],
        ids=["empty", "unparsable", "nvidia-smi failed"],
    )
    def test_undeterminable_topology_returns_empty(self, stdout: str, returncode: int) -> None:
        """Callers must skip rather than infer "no boundary present" from a parse failure."""
        with _with_topo(stdout, returncode):
            assert gpu_pairs_by_topology() == {}

    def test_missing_nvidia_smi_returns_empty(self) -> None:
        with mock.patch("isaaclab.test.utils.devices.subprocess.run", side_effect=FileNotFoundError):
            assert gpu_pairs_by_topology() == {}

    def test_row_outside_header_returns_empty(self) -> None:
        """A GPU index absent from the header must never reach CUDA_VISIBLE_DEVICES."""
        with _with_topo(_TOPO_ROW_OUTSIDE_HEADER):
            assert gpu_pairs_by_topology() == {}

    def test_truncated_matrix_returns_empty(self) -> None:
        """A partial matrix can omit exactly the rows carrying the boundary."""
        with _with_topo(_TOPO_TRUNCATED):
            assert gpu_pairs_by_topology() == {}

    def test_mig_host_returns_empty(self) -> None:
        """topo -m describes physical GPUs; CUDA addresses MIG instances."""
        with _with_topo(_TOPO_4V4, devices=_NVIDIA_SMI_L_MIG):
            assert gpu_pairs_by_topology() == {}
