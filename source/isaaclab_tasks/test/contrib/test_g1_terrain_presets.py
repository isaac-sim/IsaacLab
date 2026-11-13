"""Check the fixed-demo configuration at the same preset boundary used by the CLI."""

import pytest
from isaaclab_newton.physics.coupling import CouplerProxyCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.utils.hydra import resolve_task_config


@pytest.mark.parametrize(
    ("course", "backend", "expected_asset", "decimation"),
    (
        ("granular", "newton_mjwarp", "granular.usdc", 4),
        ("granular", "isaacsim_physx", "granular_physx.usdc", 8),
        ("granular", "physx", "granular_physx.usdc", 8),
        ("platforms", "isaacsim_physx", "platforms.usdc", 8),
        ("ramps", "newton_mjwarp", "ramps.usdc", 4),
    ),
)
def test_course_and_physics_selection_before_launch(course, backend, expected_asset, decimation):
    """Backend selection must choose compatible particles and the 50 Hz control cadence before launch."""
    cfg, _ = resolve_task_config("IsaacContrib-G1-Terrain", None, overrides=[f"presets={course}", f"physics={backend}"])
    assert cfg.scene.terrain.usd_path == f"{ISAACLAB_NUCLEUS_DIR}/Contrib/G1Terrain/{expected_asset}"
    assert cfg.decimation == cfg.sim.render_interval == decimation
    assert cfg.decimation * cfg.sim.dt == pytest.approx(0.02)
    if course != "granular":
        return
    if isinstance(cfg.sim.physics, (PhysxCfg, PhysxAutoCfg)):
        assert cfg.scene.sand is None and cfg.scene.gravel is None
    else:
        assert isinstance(cfg.sim.physics.solver_cfg, CouplerProxyCfg)
        assert cfg.scene.gravel.spawn.upper[0] == cfg.scene.sand.spawn.lower[0]
        assert cfg.scene.gravel.spawn.upper[2] == cfg.scene.sand.spawn.upper[2]
        assert cfg.sim.physics.collision_cfg.soft_contact_max == 0


def test_particle_and_gain_overrides_are_available_before_environment_construction():
    """Hydra can tune declared MPM objects and SONIC gains rather than a later hook overwriting them."""
    cfg, _ = resolve_task_config(
        "IsaacContrib-G1-Terrain",
        None,
        overrides=[
            "presets=granular",
            "physics=newton_mjwarp",
            "env.scene.sand.spawn.jitter=0.0",
            "env.scene.robot.actuators.feet.stiffness.left_ankle_pitch_joint=123.0",
        ],
    )
    assert cfg.scene.sand.spawn.jitter == 0.0
    assert cfg.scene.robot.actuators["feet"].stiffness["left_ankle_pitch_joint"] == 123.0
