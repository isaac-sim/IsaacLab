# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

RLinf VLA post-training
=======================

.. note::

   This page documents the experimental RLinf VLA post-training workflow.

Installation
------------

Follow the setup steps below from the Isaac Lab repository root.

.. code-block:: bash

   # Accept the Omniverse EULA before starting Kit-based workflows.
   # (interactive sessions prompt automatically; headless mode requires this)
   export OMNI_KIT_ACCEPT_EULA=yes

   # Step 1: Install safe dependencies via the rlinf extra
   # NOTE: On DGX Spark / aarch64 systems, build decord from source first
   # (see "Building decord on DGX Spark / aarch64" below), then run this step.
   # --inexact keeps the existing environment (e.g. Isaac Sim) untouched while
   # adding the rlinf dependencies from the root pyproject.
   uv sync --inexact --extra rlinf

   # Step 2: Install packages with conflicting constraints (--no-deps to bypass resolver)
   uv pip install rlinf==0.2.0dev2 transformers==4.51.3 "tokenizers>=0.21,<0.22" --no-deps
   # Use the official PyTorch3D v0.7.9 tag instead of the older pipablepytorch3d package.
   uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9" --no-deps

   # Step 3: Install Isaac-GR00T (pinned version)
   git clone https://github.com/NVIDIA/Isaac-GR00T.git
   cd Isaac-GR00T
   git checkout 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
   uv pip install -e ".[base]" --no-deps
   cd ../

   # Step 4: Install flash-attn (see "Skipping flash-attn" below if this fails)
   pip install flash-attn==2.8.3 --no-build-isolation --no-deps

.. _rlinf-skipping-flash-attn:

Skipping flash-attn
~~~~~~~~~~~~~~~~~~~

If ``flash-attn`` cannot be built on your platform, follow the fallback workflow described for your RLinf setup.
