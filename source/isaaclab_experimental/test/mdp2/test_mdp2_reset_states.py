# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for resetting MDP2 tasks from a bank of start states, on an analytic scene."""

from collections.abc import Collection

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.envs.reset import BetaSamplingScoreCfg, ResetStateSamplerCfg
from isaaclab.utils.warp import ProxyArray

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

pytestmark = pytest.mark.kitless

robot, target, scene_entity = mdp2.Entity("robot"), mdp2.Entity("object"), mdp2.Entity("scene")
outputs = {"joint_pos": robot.joint_pos, "object_pose_w": target.root_link_pose_w}


class Scene:
    """A robot with one joint, an object pose, and environment origins one meter apart along x."""

    def __init__(self, num_worlds: int):
        joints = mdp2.FieldSchema(("joints",), writable=True)
        self.schema = mdp2.SceneSchema(
            num_worlds=num_worlds,
            step_dt=0.02,
            entities={
                "robot": mdp2.EntitySchema(names={"joints": ("x",)}, fields={"joint_pos": joints}),
                "object": mdp2.EntitySchema(
                    names={}, fields={"root_link_pose_w": mdp2.FieldSchema((), wp.transformf, writable=True)}
                ),
                "scene": mdp2.EntitySchema(names={}, fields={"env_origins": mdp2.FieldSchema((), wp.vec3f)}),
            },
        )
        origins = np.zeros((num_worlds, 3), dtype=np.float32)
        origins[:, 0] = np.arange(num_worlds)
        identity = np.tile(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), (num_worlds, 1))
        self._data = {
            ("robot", "joint_pos"): ProxyArray(wp.zeros((num_worlds, 1), dtype=wp.float32, device=DEVICE)),
            ("object", "root_link_pose_w"): ProxyArray(wp.array(identity, dtype=wp.transformf, device=DEVICE)),
            ("scene", "env_origins"): ProxyArray(wp.array(origins, dtype=wp.vec3f, device=DEVICE)),
        }

    def data(self, entity: str, field: str) -> ProxyArray:
        return self._data[entity, field]

    def entity(self, name: str):
        return self

    def commit(self, mask: wp.array, fields: Collection[mdp2.FieldSelector]) -> None:
        """Nothing to commit: the fields are the simulation state."""


@wp.func
def _draw(world: int, joint_pos: wp.array2d(dtype=float), object_pose_w: wp.array(dtype=wp.transformf), rng: wp.int32):
    joint_pos[world, 0] = wp.randf(wp.rand_init(rng, 0), -1.0, 1.0)
    position = wp.vec3(wp.randf(wp.rand_init(rng, 1), -1.0, 1.0), 0.0, 0.0)
    object_pose_w[world] = wp.transformf(position, wp.quat_identity())


def _draw_torch(mask, joint_pos, object_pose_w, rng):
    samples = mdp2.uniform(rng, -1.0, 1.0, (2,))
    joint_pos[:, 0] = torch.where(mask, samples[:, 0], joint_pos[:, 0])
    object_pose_w[:, 0] = torch.where(mask, samples[:, 1], object_pose_w[:, 0])


draw = mdp2.Term(torch=_draw_torch, warp=_draw, random=True)


@wp.func
def _outside(world: int, joint_pos: wp.array2d(dtype=float)) -> bool:
    return wp.abs(joint_pos[world, 0]) > 0.5


outside = mdp2.Term(torch=lambda joint_pos: joint_pos[:, 0].abs() > 0.5, warp=_outside)


@wp.func
def _positive(world: int, joint_pos: wp.array2d(dtype=float)) -> float:
    return wp.where(joint_pos[world, 0] > 0.0, 1.0, 0.0)


positive = mdp2.Term(torch=lambda joint_pos: (joint_pos[:, 0] > 0.0).float(), warp=_positive)

CANDIDATES = mdp2.TaskCfg(
    events={"draw": mdp2.Event(draw, outputs=outputs)},
    terminations={"outside": mdp2.Termination(outside, inputs={"joint_pos": robot.joint_pos})},
)


def build(scene: Scene, backend: str = "warp", **params) -> tuple[mdp2.MDP, mdp2.MDPState]:
    """Build a task whose only reset event restores from a bank, credited by whether the joint ended positive."""
    inputs = params.pop("inputs", {"succeeded": mdp2.quantities.success})
    task = mdp2.TaskCfg(
        quantities={"success": mdp2.Quantity(positive, inputs={"joint_pos": robot.joint_pos})},
        events={"reset": mdp2.Event(mdp.reset_from_bank, inputs=inputs, outputs=outputs, params=params)},
    )
    builder = mdp2.MDPBuilder(scene.schema)
    builder.add_task(task)
    task_mdp = builder.finalize(device=DEVICE, backend=backend)
    state = task_mdp.state(scene, seed=3)
    mdp2.startup(task_mdp, state)
    return task_mdp, state


def reset(task_mdp, state, mask) -> None:
    mdp2.reset(task_mdp, state, ProxyArray(wp.from_torch(torch.tensor(mask, device=DEVICE))))


def joint_pos(scene: Scene) -> torch.Tensor:
    return scene.data("robot", "joint_pos").torch[:, 0]


RECORDING = {"succeeded": mdp2.quantities.success, "terminated": mdp2.terminated}


def test_candidates_fill_each_world_with_valid_states_and_restore_only_reset_worlds():
    scene = Scene(num_worlds=8)
    task_mdp, state = build(scene, fill=mdp.FillFromCandidates(CANDIDATES, bank_size=4))
    bank = mdp.reset_bank(state)

    banked = bank.states["joint_pos"][..., 0].cpu()
    assert banked.shape == (8, 4) and (banked.abs() <= 0.5).all() and len(banked.unique()) == 32

    reset(task_mdp, state, [True] * 8)
    slots = bank.slots.cpu()
    assert torch.equal(joint_pos(scene).cpu(), banked[torch.arange(8), slots])  # each world draws its own bank

    before = joint_pos(scene).clone()
    for _ in range(8):
        reset(task_mdp, state, [True] + [False] * 7)
    assert torch.equal(joint_pos(scene)[1:], before[1:])


def test_shared_bank_restores_poses_relative_to_each_world_origin():
    scene = Scene(num_worlds=4)
    poses = torch.tensor([[0.1, 0.2, 0.3, 0, 0, 0, 1], [-0.1, 0.0, 0.5, 0, 0, 0, 1]], device=DEVICE)
    states = {"joint_pos": torch.tensor([[0.25], [-0.25]], device=DEVICE), "object_pose_w": poses}
    task_mdp, state = build(
        scene,
        fill=mdp.FillFromStates(states),
        shared=True,
        inputs={"succeeded": mdp2.quantities.success, "env_origins": scene_entity.env_origins},
    )
    reset(task_mdp, state, [True] * 4)

    slot = mdp.reset_bank(state).slots
    origins = scene.data("scene", "env_origins").torch
    torch.testing.assert_close(scene.data("object", "root_link_pose_w").torch[:, :3], poses[slot, :3] + origins)
    torch.testing.assert_close(scene.data("robot", "joint_pos").torch, states["joint_pos"][slot])


def test_outcomes_are_credited_to_the_state_each_episode_started_from():
    scene = Scene(num_worlds=16)
    states = {
        "joint_pos": torch.tensor([[0.5], [-0.5]], device=DEVICE),
        "object_pose_w": torch.tensor([[0, 0, 0, 0, 0, 0, 1]] * 2, dtype=torch.float32, device=DEVICE),
    }
    sampler = ResetStateSamplerCfg(history_length=8, scores=[BetaSamplingScoreCfg(target=1.0)])
    inputs = {"succeeded": mdp2.quantities.success, "env_origins": scene_entity.env_origins}
    task_mdp, state = build(scene, fill=mdp.FillFromStates(states), shared=True, sampler=sampler, inputs=inputs)
    transition = task_mdp.transition()
    for _ in range(6):
        reset(task_mdp, state, [True] * 16)
        mdp2.evaluate(task_mdp, state, transition)  # success: the restored joint position is positive

    bank = mdp.reset_bank(state)
    np.testing.assert_allclose(bank.success_rate.cpu().numpy(), [1.0, 0.0])
    assert bank.episode_counts.sum() == 5 * 16  # the first reset has no episode to credit


def test_recording_banks_visited_states_that_replay_restores():
    scene = Scene(num_worlds=2)
    sampler = ResetStateSamplerCfg(history_length=4, scores=[BetaSamplingScoreCfg(target=0.5)])
    record = mdp.RecordStates(slots=2, probability=1.0)
    inputs = {**RECORDING, "record_when": mdp2.quantities.success}
    fill = mdp.FillFromCandidates(CANDIDATES, bank_size=2)
    task_mdp, state = build(scene, fill=fill, sampler=sampler, record=record, inputs=inputs)
    bank, transition = mdp.reset_bank(state), task_mdp.transition()
    filled = bank.states["joint_pos"][:, :2].clone()
    assert torch.equal(bank.states["joint_pos"][:, 2:], filled)  # recorded slots start as copies

    # Only world 1 visits a successful state, which is recorded into its first recorded slot with a fresh prior.
    joint_pos(scene).copy_(torch.tensor([-0.3, 0.7], device=DEVICE))
    mdp2.evaluate(task_mdp, state, transition)
    banked = bank.states["joint_pos"][..., 0]
    assert torch.equal(banked[0], filled[0, :, 0].repeat(2)) and banked[1, 2] == 0.7
    assert bank.episode_counts[1, 2] == 0 and bank.success_rate[1, 2] == 0.5

    # Replay restarts world 1 from the recorded state; world 0 draws from its own bank.
    bank.replay(torch.tensor([-1, 2]))
    joint_pos(scene).fill_(9.0)
    reset(task_mdp, state, [True, True])
    assert joint_pos(scene)[1] == 0.7 and joint_pos(scene)[0] in filled[0]
    assert bank.slots.tolist()[1] == 2

    # Recording over the slot the episode started from drops that episode's credit.
    joint_pos(scene).copy_(torch.tensor([0.1, 0.2], device=DEVICE))
    mdp2.evaluate(task_mdp, state, transition)  # world 1 records into slot 3
    mdp2.evaluate(task_mdp, state, transition)  # then over slot 2
    bank.replay(None)
    reset(task_mdp, state, [False, True])
    assert bank.episode_counts[1, 2] == 0 and (bank.states["joint_pos"][1, 2:, 0] == 0.2).all()


def test_nominal_resets_keep_the_earlier_events_and_are_not_credited():
    scene = Scene(num_worlds=4)
    task_mdp, state = build(scene, fill=mdp.FillFromCandidates(CANDIDATES, bank_size=8), nominal_probability=1.0)
    bank = mdp.reset_bank(state)
    joint_pos(scene).fill_(9.0)
    for _ in range(2):
        reset(task_mdp, state, [True] * 4)
    assert (joint_pos(scene) == 9.0).all() and (bank.slots == -1).all() and bank.episode_counts.sum() == 0

    bank.replay(torch.tensor([5, -1, -1, -1]))  # a replay request overrides the nominal draw
    reset(task_mdp, state, [True] * 4)
    assert joint_pos(scene)[0] == bank.states["joint_pos"][0, 5, 0] and (joint_pos(scene)[1:] == 9.0).all()


def test_saved_banks_fill_later_runs_with_the_same_worlds(tmp_path):
    scene = Scene(num_worlds=4)
    _, state = build(scene, fill=mdp.FillFromCandidates(CANDIDATES, bank_size=3))
    path = tmp_path / "bank.pt"
    mdp.reset_bank(state).save(path)

    _, loaded = build(Scene(num_worlds=4), fill=mdp.FillFromFile(path))
    for name, value in mdp.reset_bank(state).states.items():
        assert torch.equal(mdp.reset_bank(loaded).states[name], value)
    with pytest.raises(ValueError, match="holds banks of 4 worlds"):
        build(Scene(num_worlds=2), fill=mdp.FillFromFile(path))


def test_backends_and_graph_replay_restore_and_record_the_same_states():
    runs = []
    for backend, capture in (("torch", False), ("warp", False), ("warp", True)):
        if capture and not DEVICE.startswith("cuda"):
            continue
        scene = Scene(num_worlds=32)
        task_mdp, state = build(
            scene,
            backend,
            fill=mdp.FillFromCandidates(CANDIDATES, bank_size=8),
            sampler=ResetStateSamplerCfg(history_length=4, scores=[BetaSamplingScoreCfg(target=0.5)]),
            nominal_probability=0.25,
            record=mdp.RecordStates(slots=4, probability=0.5),
            inputs=RECORDING,
        )
        bank = mdp.reset_bank(state)
        mask = ProxyArray(wp.zeros(32, dtype=wp.bool, device=DEVICE))
        transition = task_mdp.transition()
        actions = ProxyArray(wp.zeros((32, 0), dtype=wp.float32, device=DEVICE))
        captured = mdp2.CapturedMDP(task_mdp, state, transition, actions, mask) if capture else None
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        history = []
        for step in range(12):
            mask.torch.copy_(torch.rand(32, device=DEVICE, generator=generator) < 0.5)
            joint_pos(scene).add_(0.01)  # the worlds move, so that records differ from the filled states
            bank.replay(torch.arange(32, device=DEVICE) % 12 - 4 if step == 6 else None)
            if captured is None:
                mdp2.reset(task_mdp, state, mask)
                mdp2.evaluate(task_mdp, state, transition)
            else:
                captured.reset()
                captured.evaluate()
            history.append(torch.cat([joint_pos(scene), bank.states["joint_pos"].flatten()]))
        runs.append(torch.stack(history))
    assert not torch.equal(runs[0][-1, 32:], runs[0][0, 32:])  # the worlds recorded states
    for run in runs[1:]:
        assert torch.equal(run, runs[0])


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"fill": mdp.FillFromStates({"joint_pos": torch.zeros(2, 1, 1)})}, "but the event writes"),
        ({"fill": mdp.FillFromCandidates(CANDIDATES, bank_size=2), "shared": True}, "needs the env_origins input"),
        (
            {"fill": mdp.FillFromCandidates(CANDIDATES, bank_size=2), "record": mdp.RecordStates(2, 0.1)},
            "the terminated input",
        ),
    ],
    ids=["fill_misses_an_output", "shared_poses_without_origins", "recording_without_terminated"],
)
def test_invalid_banks_are_rejected_at_startup(params, message):
    with pytest.raises(ValueError, match=message):
        build(Scene(num_worlds=2), **params)
