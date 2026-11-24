# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the physics presets that run a task on a model, and for the settings of the models.

Nothing here starts a simulation, and only the test that builds the recipes needs ``newton-neural``.
"""

import dataclasses
import subprocess
import sys
import textwrap

import pytest
from isaaclab_neural import NeRDRecipeCfg
from isaaclab_neural.physics import MJWarpNeRDSolverCfg, NeRDSolverCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.managers import EventTermCfg
from isaaclab.utils import instantiate, validate

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_task_config
from isaaclab_tasks.utils.hydra import PresetTarget, enumerate_task_presets, resolve_task_runs

from nerd_test_utils import ALLEGRO_REORIENT, ANYMAL_C_FLAT, FRANKA_REACH, TASKS, nerd_preset  # isort: skip

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

AGENT = "rsl_rl_cfg_entry_point"
RECIPE = "nerd_recipe_cfg_entry_point"
FRANKA_REACH_OSC = "Isaac-Reach-Franka-OSC"
ANYMAL_C_ROUGH = "IsaacContrib-Velocity-Rough-AnymalC"
ANYMAL_C_NAVIGATION = "IsaacContrib-Navigation-Flat-AnymalC"
SOLVER_CFG_TYPES = {
    FRANKA_REACH: NeRDSolverCfg,
    ANYMAL_C_FLAT: NeRDSolverCfg,
    ALLEGRO_REORIENT: MJWarpNeRDSolverCfg,
}
"""Solver configuration of the model preset of each task."""
RECIPE_NUMBERS = {
    FRANKA_REACH: ("state_speed", 2.5),
    ANYMAL_C_FLAT: ("drop", 0.4),
    ALLEGRO_REORIENT: ("constraint_count", 128),
}
"""A number of the recipe of each task, with a value other than its default."""
UNSEEN_EVENTS = {ANYMAL_C_FLAT: ("add_base_mass", "base_com")}
"""Events that the recipe of a task names as unseen by its model. The other tasks name none."""


@pytest.mark.parametrize("task", TASKS)
def test_nerd_preset_keeps_the_task(task):
    """A model preset keeps the timing of the task, names its robot and logs its policies apart."""
    preset = nerd_preset(task)
    mujoco, mujoco_agent = resolve_task_config(task, AGENT, overrides=("physics=newton_mjwarp",))
    env_cfg, agent_cfg = resolve_task_config(task, AGENT, overrides=(f"physics={preset}",))
    sim, physics, solver_cfg = env_cfg.sim, env_cfg.sim.physics, env_cfg.sim.physics.solver_cfg

    assert type(mujoco.sim.physics.solver_cfg) is MJWarpSolverCfg
    timing = (sim.dt, env_cfg.decimation, sim.render_interval)
    assert timing == (mujoco.sim.dt, mujoco.decimation, mujoco.sim.render_interval)
    assert type(physics) is NewtonCfg
    assert type(solver_cfg) is SOLVER_CFG_TYPES[task]
    if isinstance(solver_cfg, NeRDSolverCfg):
        # A model in place of the solver takes one solver step per environment step, under a manager of its own.
        assert physics.num_substeps == 1
        assert physics.class_type != mujoco.sim.physics.class_type
    else:
        # A model inside MuJoCo Warp takes the solver steps of the task, under the manager of the task.
        assert physics.num_substeps == mujoco.sim.physics.num_substeps
        assert physics.class_type == mujoco.sim.physics.class_type
    assert solver_cfg.articulation == env_cfg.scene.robot.prim_path
    assert agent_cfg.experiment_name != mujoco_agent.experiment_name


def test_task_timing_is_plain_numbers():
    """The timing of a task is a number in its configuration class, so other tasks can compute with it."""
    for task in TASKS:
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        assert isinstance(cfg.decimation, int), task
        assert isinstance(cfg.sim.dt, float), task
        assert isinstance(cfg.sim.render_interval, int), task
    # The navigation task multiplies the decimation of the ANYmal C flat task.
    walking, _ = resolve_task_config(ANYMAL_C_FLAT, None, overrides=())
    navigation, _ = resolve_task_config(ANYMAL_C_NAVIGATION, None, overrides=())
    assert navigation.decimation % walking.decimation == 0
    assert navigation.sim.dt == walking.sim.dt


def test_tasks_built_on_a_model_task_offer_no_model():
    """A task that builds on a task with a model does not offer that model, which stands for the other task."""
    # The navigation task steps the walking robot with a policy of its own and takes that task as it is by default.
    preset = nerd_preset(ANYMAL_C_FLAT)
    offered = enumerate_task_presets(ANYMAL_C_NAVIGATION)
    assert all(preset not in names for names in offered.values())
    # The operational space controller replaces the joint drives that the model of the arm stands for. The task
    # offers every other physics of the joint position task.
    preset = nerd_preset(FRANKA_REACH)
    physics = set(enumerate_task_presets(FRANKA_REACH)[PresetTarget.PHYSICS])
    offered = enumerate_task_presets(FRANKA_REACH_OSC)
    assert set(offered[PresetTarget.PHYSICS]) == physics - {preset}
    assert all(preset not in names for names in offered.values())


@pytest.mark.parametrize("task", TASKS)
def test_missing_checkpoint_names_the_scripts(task, tmp_path, monkeypatch):
    """Without a trained model its preset names the scripts that make one, and the other presets are not affected."""
    monkeypatch.chdir(tmp_path)
    preset = nerd_preset(task)
    env_cfg, _ = resolve_task_config(task, AGENT, overrides=(f"physics={preset}",))
    with pytest.raises(ValueError, match=r"scripts/neural/record\.py and scripts/neural/train\.py"):
        validate(env_cfg)
    validate(resolve_task_config(task, AGENT, overrides=("physics=newton_mjwarp",))[0])

    # The tools that list what a task can run leave the preset out, and keep offering its name.
    assert preset in enumerate_task_presets(task)[PresetTarget.PHYSICS]
    runs = resolve_task_runs(task)
    assert runs and all(physics != preset for physics, _, _ in runs)

    # The command line names another file, a precision and a step without compilation.
    checkpoint = tmp_path / "elsewhere.pt"
    checkpoint.touch()
    solver = "env.sim.physics.solver_cfg"
    overrides = (f"{solver}.checkpoint={checkpoint}", f"{solver}.precision=bf16", f"{solver}.compile=false")
    env_cfg, _ = resolve_task_config(task, AGENT, overrides=(f"physics={preset}", *overrides))
    solver_cfg = env_cfg.sim.physics.solver_cfg
    assert (solver_cfg.checkpoint, solver_cfg.precision, solver_cfg.compile) == (str(checkpoint), "bf16", False)
    validate(env_cfg)


def test_franka_model_requires_joint_position_actions(tmp_path):
    """The model of the Franka arm stands for joint position actions with the drives it was recorded with."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    preset = nerd_preset(FRANKA_REACH)
    overrides = (f"physics={preset}", f"env.sim.physics.solver_cfg.checkpoint={checkpoint}")
    env_cfg, _ = resolve_task_config(FRANKA_REACH, AGENT, overrides=(*overrides, "presets=diffik"))
    with pytest.raises(ValueError, match=f"'{preset}' physics preset requires the Franka joint-position action"):
        validate(env_cfg)
    # The hold preset softens the drives. No agent is resolved, since the RSL-RL agent names its own conflict.
    env_cfg, _ = resolve_task_config(FRANKA_REACH, None, overrides=(*overrides, "presets=hold"))
    with pytest.raises(ValueError, match=f"'{preset}' physics preset does not support the 'hold' preset"):
        validate(env_cfg)


def test_anymal_model_clips_its_joint_targets():
    """Under its model the walking task keeps the joint targets within the range of the recorded ones."""
    reach = load_cfg_from_registry(ANYMAL_C_FLAT, RECIPE).target_reach
    env_cfg, agent_cfg = resolve_task_config(ANYMAL_C_FLAT, AGENT, overrides=(f"physics={nerd_preset(ANYMAL_C_FLAT)}",))
    joint_pos, pose = env_cfg.actions.joint_pos, env_cfg.scene.robot.init_state.joint_pos
    # Every joint of the default pose has a range [rad] around its angle, as wide as the recipe spreads the targets.
    assert joint_pos.clip.keys() == pose.keys()
    for joints, (low, high) in joint_pos.clip.items():
        assert ((low + high) / 2, (high - low) / 2) == pytest.approx((pose[joints], reach)), joints
    # The RSL-RL agent clips the actions to the same range, so the policy observes the action that was applied.
    assert agent_cfg.clip_actions * joint_pos.scale == pytest.approx(reach)

    # On a solver the task takes any target, and so do the rough task that it extends and the navigation task
    # that steps it.
    for task in (ANYMAL_C_FLAT, ANYMAL_C_ROUGH):
        solver, _ = resolve_task_config(task, None, overrides=("physics=newton_mjwarp",))
        assert solver.actions.joint_pos.clip is None, task
    navigation, _ = resolve_task_config(ANYMAL_C_NAVIGATION, None, overrides=())
    assert navigation.actions.pre_trained_policy_action.low_level_actions.clip is None


def test_allegro_model_keeps_the_nominal_hand():
    """Under its model the Allegro task randomizes the cube and leaves the mass and the gains of the hand alone."""
    events = {}
    for physics in ("newton_mjwarp", nerd_preset(ALLEGRO_REORIENT)):
        overrides = (f"physics={physics}", "presets=randomized")
        env_cfg, _ = resolve_task_config(ALLEGRO_REORIENT, None, overrides=overrides)
        events[physics] = {name for name, term in vars(env_cfg.events).items() if term is not None}
    reference, model = events.values()
    # The model stands for the hand it was recorded with, and MuJoCo Warp still sees the cube.
    assert model == reference - {"robot_scale_mass", "robot_joint_stiffness_and_damping"}
    assert len(reference) == len(model) + 2
    assert {"object_scale_mass", "object_physics_material"} <= model


@pytest.mark.parametrize("task", TASKS)
def test_recipe_settings(task):
    """Each task registers the settings of its model, and the command line reaches their numbers."""
    recipe_cfg = load_cfg_from_registry(task, RECIPE)
    assert isinstance(recipe_cfg, NeRDRecipeCfg)
    assert recipe_cfg.physics_preset == nerd_preset(task)
    assert recipe_cfg.reference_preset == "newton_mjwarp"
    reference, _ = resolve_task_config(task, None, overrides=(f"physics={recipe_cfg.reference_preset}",))
    assert hasattr(reference.scene, recipe_cfg.robot)
    # The events that randomize what the model does not see are events of the task, by their names.
    assert tuple(recipe_cfg.unseen_events) == UNSEEN_EVENTS.get(task, ())
    for name in recipe_cfg.unseen_events:
        assert isinstance(getattr(reference.events, name), EventTermCfg), name

    # The scripts resolve the settings as an agent, so a number of the task and the amount of data are set as
    # ``agent.<path>=value``.
    name, value = RECIPE_NUMBERS[task]
    assert getattr(recipe_cfg, name) != value
    overrides = (f"agent.{name}={value}", "agent.record.episode_frames=[8,16]")
    _, changed = resolve_task_config(task, RECIPE, overrides=overrides)
    assert type(changed) is type(recipe_cfg)
    assert (getattr(changed, name), tuple(changed.record.episode_frames)) == (value, (8, 16))


@pytest.mark.parametrize("task", TASKS)
def test_recipe_class(task):
    """The settings of a task build the recipe of that task, which keeps them."""
    pytest.importorskip("newton_neural", reason="needs the newton-neural extra")
    from isaaclab_neural import NeRDRecipe  # noqa: PLC0415

    recipe_cfg = load_cfg_from_registry(task, RECIPE)
    recipe = instantiate(recipe_cfg)
    assert isinstance(recipe, NeRDRecipe)
    assert type(recipe) is not NeRDRecipe
    # The recipe lies beside its settings, in the folder of the task.
    assert type(recipe).__module__.rpartition(".")[0] == type(recipe_cfg).__module__.rpartition(".")[0]
    assert recipe.cfg is recipe_cfg


def test_from_mjwarp():
    """The solver of a model inside MuJoCo Warp copies the settings of MuJoCo Warp and keeps its own solver class."""
    mujoco = resolve_task_config(ALLEGRO_REORIENT, None, overrides=("physics=newton_mjwarp",))[0].sim.physics.solver_cfg
    assert type(mujoco) is MJWarpSolverCfg
    copied = [field.name for field in dataclasses.fields(MJWarpSolverCfg) if field.name != "class_type"]
    # Every setting gets a value of its own, so that one left at its default cannot pass for copied. Isaac Lab
    # itself turns ls_parallel off, so that one stays as it is.
    for name in copied:
        if name != "ls_parallel":
            setattr(mujoco, name, f"{name} of the task")
    solver_cfg = MJWarpNeRDSolverCfg.from_mjwarp(mujoco, checkpoint="hand.pt", articulation="/World/Robot", njmax=7)

    for name in copied:
        assert getattr(solver_cfg, name) == (7 if name == "njmax" else getattr(mujoco, name)), name
    assert solver_cfg.class_type == MJWarpNeRDSolverCfg().class_type
    assert solver_cfg.class_type != mujoco.class_type
    assert (solver_cfg.checkpoint, solver_cfg.articulation) == ("hand.pt", "/World/Robot")


def test_configurations_load_without_newton_neural(tmp_path):
    """Resolve every model preset and every recipe settings in a process where ``newton-neural`` cannot be imported.

    Tasks import the physics and solver configurations when their modules load, and tools read the settings of a
    recipe, so these must not need the library. It loads when the physics of a task or a recipe is built, where a
    missing library is Python's own import error.
    """
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    script = textwrap.dedent(f"""
        import sys


        class Uninstalled:
            # Fails every import of the library as an environment without it does.
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "newton_neural":
                    raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)


        sys.meta_path.insert(0, Uninstalled())

        import pytest
        import isaaclab_neural
        from isaaclab.app import validate_launch
        from isaaclab.utils import instantiate, validate
        from isaaclab_tasks.utils import resolve_task_config

        solver_cfgs, recipe_cfgs = [], []
        for task, preset in {[(task, nerd_preset(task)) for task in TASKS]!r}:
            overrides = (f"physics={{preset}}", "env.sim.physics.solver_cfg.checkpoint={checkpoint}")
            env_cfg, _ = resolve_task_config(task, "{AGENT}", overrides=overrides)
            validate(env_cfg)
            validate_launch(env_cfg, {{"livestream": 0}})
            solver_cfgs.append(env_cfg.sim.physics.solver_cfg)
            recipe_cfgs.append(resolve_task_config(task, "{RECIPE}", overrides=overrides)[1])

        # Of the library, of Newton and of the adapters nothing is loaded, only configuration modules.
        loaded = [name for name in sys.modules if name.split(".")[0] in ("newton_neural", "newton", "mujoco_warp")]
        loaded += [name for name in sys.modules if name.startswith("isaaclab_neural.") or ".nerd_recipe" in name]
        assert [name for name in loaded if not name.endswith(("_cfg", "isaaclab_neural.physics"))] == [], loaded

        # The solver class and the recipe are where the library is first needed.
        for solver_cfg in solver_cfgs:
            with pytest.raises(ModuleNotFoundError, match="^No module named 'newton_neural'$"):
                solver_cfg.class_type.solver_class
        for recipe_cfg in recipe_cfgs:
            with pytest.raises(ModuleNotFoundError, match="^No module named 'newton_neural'$"):
                instantiate(recipe_cfg)
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]
