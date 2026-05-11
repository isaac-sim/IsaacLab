# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single source of truth for shared external pip dependencies.

Every ``setup.py`` in ``source/`` that needs one of these packages imports the
constant from here so a version bump touches one line. The wheel builder's
``tools/wheel_builder/res/python_packages.toml`` mirrors these values and is
kept in sync by :mod:`tools.test_python_deps_sync`.
"""

# NOTE: keep this file dependency-free (standard library only) so it can be
# imported by ``setup.py`` before pip has resolved anything.

NEWTON = "newton[sim] @ git+https://github.com/newton-physics/newton.git@v1.2.0rc2"
"""Newton physics engine, pulled via the ``sim`` extra so ``mujoco`` /
``mujoco-warp`` come along transitively. Every ``newton @ ...`` declaration
in the repo must use this exact spec — pip resolves a git-URL requirement
once per URL, so a bare declaration anywhere shadows requested extras
elsewhere."""

WARP_LANG = "warp-lang==1.13.0"
"""NVIDIA Warp Python bindings."""

TORCH = "torch>=2.10"
"""PyTorch floor for CUDA 12.8/13 wheel compatibility."""

NUMPY = "numpy>=2"
"""NumPy 2.x is required for the rest of the stack."""

PRETTYTABLE = "prettytable==3.3.0"
"""Tabular formatting; pinned to avoid Isaac Sim prebundle conflicts."""

PYOPENGL_ACCELERATE = "PyOpenGL-accelerate==3.1.10"
"""PyOpenGL Cython accelerator; pinned across Newton-side packages."""
