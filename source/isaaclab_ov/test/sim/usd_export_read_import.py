# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load an exported USD file into a new OVPhysX instance and read its objects, for the USD export test.

Run it in a separate process: a process has one attached OVPhysX simulation.
"""


def read_import(source: str, articulations: list[str], bodies: list[str]) -> dict:
    """Import a USD layer string into a new OVPhysX instance and read its physics objects.

    Args:
        source: USD layer contents.
        articulations: Articulation root paths.
        bodies: Rigid bodies outside articulations.

    Returns:
        Raw readings as returned by :func:`isaaclab_ov.sim.usd_export.read`.
    """
    import ovstage
    from isaaclab_ov.physics import OvPhysxManager
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxBackend
    from isaaclab_ov.physics.ovphysx_manager_cfg import OvPhysxBackendCfg
    from isaaclab_ov.sim.usd_export.usd_exporter import read
    from isaaclab_ov.stage import create_ovstage

    OvPhysxManager._ensure_physx_schemas_registered()
    backend = OvPhysxBackend(OvPhysxBackendCfg(device="cpu"))
    try:
        backend.stage = create_ovstage("usd_export_test")
        ovstage.population.open_usd_from_string(backend.stage, source, ordinal=1, domains=ovstage.PopulationDomain.ALL)
        backend.stage.advance_write_floor(ordinal=1).wait()
        backend.physx.attach_ovstage(backend.stage, read_ordinal=1)
        return read(backend.physx, articulations, bodies, "cpu")
    finally:
        backend.close()
