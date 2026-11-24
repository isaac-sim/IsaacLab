# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the library-owned neural dynamics train workflow."""

import warp as wp

wp.config.enable_backward = False

from isaaclab_neural.entrypoints.train import main  # noqa: E402

if __name__ == "__main__":
    main()
