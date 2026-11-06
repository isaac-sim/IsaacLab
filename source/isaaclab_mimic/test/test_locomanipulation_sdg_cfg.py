# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The SDG task configuration can be loaded before starting its simulation runtime."""

import subprocess
import sys
import textwrap


def test_sdg_config_loads_before_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""\
                import sys
                import isaaclab_mimic.locomanipulation_sdg.envs
                from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
                from isaaclab_tasks.utils import parse_env_cfg

                sys.modules["isaaclab_mimic.locomanipulation_sdg.envs.g1_locomanipulation_sdg_env"] = None
                cfg = parse_env_cfg("Isaac-G1-SteeringWheel-Locomanipulation", device="cpu", num_envs=1)
                cfg.scene.add_forklifts(1)
                cfg.scene.add_boxes(1)
                assert cfg.scene.forklift_0.prim_path == "{ENV_REGEX_NS}/Forklift0"
                assert cfg.scene.box_0.prim_path == "{ENV_REGEX_NS}/Box0"
                assert cfg.scene.num_envs == 1
                assert cfg.sim.device == "cpu"
                assert cfg.actions.upper_body_ik.controller.urdf_path.startswith(ISAACLAB_NUCLEUS_DIR + "/")
                assert not {"pxr", "omni", "carb", "isaacsim", "scipy"}.intersection(sys.modules)
            """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
