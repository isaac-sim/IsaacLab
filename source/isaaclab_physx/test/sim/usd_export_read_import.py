# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import a USD file into a headless Isaac Sim and pickle its PhysX readings.

The USD export test loads exported files with this script, in a separate process because a process
has one attached PhysX simulation::

    python usd_export_read_import.py <stage.usda> <output.pkl> '[[articulations], [bodies]]'
"""

import json
import pickle
import sys


def main(source: str, output: str, articulations: list[str], bodies: list[str]):
    """Import ``source`` and write the readings of the given objects to ``output``."""
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        from isaaclab_physx.sim.usd_export.usd_exporter import read

        import omni.physics.tensors
        import omni.physx
        import omni.usd

        context = omni.usd.get_context()
        if not context.open_stage(source):
            raise RuntimeError(f"Could not open '{source}'.")
        physx = omni.physx.get_physx_interface()
        physx.force_load_physics_from_usd()
        physx.start_simulation()
        view = omni.physics.tensors.create_simulation_view("numpy", stage_id=context.get_stage_id())
        view.set_subspace_roots("/")
        with open(output, "wb") as file:
            pickle.dump(read(view, articulations, bodies), file)
    finally:
        app.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *json.loads(sys.argv[3]))
