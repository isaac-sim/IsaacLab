# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 reset viewer on the Kuka Allegro lift task, on Newton."""

import dataclasses
from collections.abc import Iterator

import pytest
import torch
from isaaclab_experimental.envs import MDP2RLEnv, ResetViewer
from isaaclab_experimental.envs.mdp2_reset_viewer import ViserResetPanel

from isaaclab_tasks.mdp2.lift_kuka_allegro.lift_env_cfg import KukaAllegroLiftEnvCfg, KukaAllegroLiftPhysicsCfg

pytestmark = [pytest.mark.kitless, pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")]

NUM_ENVS = 8


@pytest.fixture(scope="module")
def viewer() -> Iterator[ResetViewer]:
    cfg = KukaAllegroLiftEnvCfg()
    cfg.sim.physics = KukaAllegroLiftPhysicsCfg().newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    params = cfg.task.events["reset"].params
    params["fill"] = dataclasses.replace(params["fill"], bank_size=3)
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        yield ResetViewer(env)
    finally:
        env.close()


def test_viewing_restores_slots_rolls_candidates_and_credits_nothing(viewer: ResetViewer):
    env, bank = viewer.env, viewer.bank
    viewer.request("show", 1)
    viewer.request("select", 3)
    viewer.update()
    assert viewer.snapshot.mode == "slot" and viewer.snapshot.world == 3
    assert (bank.slots == 1).all()
    torch.testing.assert_close(env.scene["robot"].data.joint_pos.torch, bank.states["joint_pos"][:, 1])

    viewer.roll()
    accepted = ~viewer._candidates.transition.terminated.torch
    labels = ["accepted" if flag else "rejected" for flag in accepted.tolist()]
    viewer.update()
    assert viewer.snapshot.mode == "candidates" and viewer.snapshot.labels == labels

    # Candidates are not episodes: playing first draws starts as training does, and nothing is credited.
    viewer.play()
    assert viewer.mode == "draw" and viewer.playing
    for _ in range(3):
        viewer.update()
    assert bank.episode_counts.sum() == 0
    viewer.show(0)
    assert not viewer.playing and (bank.slots == 0).all() and bank.episode_counts.sum() == 0


def test_viser_panel_shows_the_bank_and_labels_every_world(viewer: ResetViewer):
    viser = pytest.importorskip("viser")
    server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    try:
        panel = ViserResetPanel(viewer, server)
        viewer.show(2)
        viewer._panels.append(panel)
        viewer.update()
        assert [label.text for label in panel._labels] == viewer.snapshot.labels
        assert ">2 *</td>" in panel._table.content
        assert "slot 2" in panel._status.content and panel._slot.value == 2
    finally:
        viewer._panels.clear()
        server.stop()
