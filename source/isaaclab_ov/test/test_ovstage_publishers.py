# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVStage publishers write scene data provider state once per change for every consumer of a stage."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

pytest.importorskip("ovstage")

import isaaclab_ov.stage_publishers as stage_publishers  # noqa: E402
import ovstage  # noqa: E402
from isaaclab_ov.stage import OvstageBackend, OvstageBackendCfg  # noqa: E402
from isaaclab_ov.stage_publishers import BodyPosePublisher  # noqa: E402

from pxr import Gf, Usd, UsdGeom  # noqa: E402

from isaaclab.scene_data import SceneDataFormat, SceneDataProvider  # noqa: E402

pytestmark = [pytest.mark.kitless, pytest.mark.ov]

_ENV_PATHS = ["/World/envs/env_0", "/World/envs/env_1"]
_BODY_PATHS = [f"{env}/Body" for env in _ENV_PATHS]


def _physics(device: str = "cpu") -> tuple[SimpleNamespace, SceneDataFormat.Transform]:
    """Return a physics backend publishing one body per environment, and its mutable poses."""
    transforms = SceneDataFormat.Transform()
    transforms.transforms = wp.zeros(len(_BODY_PATHS), dtype=wp.transformf, device=device)
    backend = SimpleNamespace(
        transform_count=len(_BODY_PATHS),
        transform_paths=_BODY_PATHS,
        transforms_timestamp=0,
        get_transforms=lambda _format: transforms,
    )
    return backend, transforms


def _set_positions(backend: SimpleNamespace, transforms: SceneDataFormat.Transform, position) -> None:
    """Publish the same position for every body under a new producer timestamp."""
    poses = np.array([[*position, 0, 0, 0, 1]] * len(_BODY_PATHS), dtype=np.float32)
    transforms.transforms.assign(poses)
    backend.transforms_timestamp += 1


def _recording_scene(monkeypatch) -> tuple[OvstageBackend, list]:
    """Return a stage shell that records native writes and resource releases in order."""
    monkeypatch.setattr(stage_publishers, "xform_tensor_from_warp", lambda matrices: matrices)
    events = []
    scene = OvstageBackend.__new__(OvstageBackend)
    scene._publishers = {}
    scene.resources = contextlib.ExitStack()
    scene.resources.callback(events.append, ("close_stage",))
    scene.clone_copies = []
    scene.clone_env_paths = _ENV_PATHS
    scene.ordinal = 3
    scene.paths = SimpleNamespace(
        create_path_list_from_strings=list, destroy_path_list=lambda paths: events.append(("destroy_paths", paths))
    )
    scene.stage = SimpleNamespace(
        query_from_path_list=contextlib.nullcontext,
        write_attribute=lambda query, name, **kwargs: (
            events.append(("write", name, kwargs)) or SimpleNamespace(wait=lambda: None)
        ),
    )
    return scene, events


def test_consumers_of_a_shared_stage_publish_one_pose_set_per_change(monkeypatch):
    """Every consumer reaches one publisher, and visibility is stage state rather than a consumer filter."""
    scene, events = _recording_scene(monkeypatch)
    backend, transforms = _physics()
    provider = SceneDataProvider(backend)
    usd = Usd.Stage.CreateInMemory()

    def create():
        return BodyPosePublisher(scene, provider, usd, "cpu")

    publisher = scene.publisher(BodyPosePublisher, create)
    assert scene.publisher(BodyPosePublisher, create) is publisher  # A second renderer shares the publisher.
    assert [event[1] for event in events if event[0] == "write"] == ["omni:resetXformStack"]
    events.clear()

    _set_positions(backend, transforms, (1, 2, 3))
    scene.publish()
    scene.publish()  # The second consumer's refresh of the same physics state writes nothing.
    assert [event[1] for event in events] == ["omni:xform"]

    # Hiding an environment while physics is paused republishes, whichever consumer refreshes next.
    publisher.set_hidden_environments([False, True])
    scene.publish()
    scene.publish()
    assert len(events) == 2
    write = events[-1][2]
    np.testing.assert_allclose(write["tensors"].numpy()[:, 3, :3], [[1, 2, 3], [1, 2, 3 - 1.0e6]])
    assert write["ordinal"] == 3
    assert "cuda_stream" not in write  # Host poses need no stream ordering.

    # Later physics steps keep the hidden environment hidden for every consumer.
    _set_positions(backend, transforms, (4, 5, 6))
    scene.publish()
    np.testing.assert_allclose(events[-1][2]["tensors"].numpy()[:, 3, :3], [[4, 5, 6], [4, 5, 6 - 1.0e6]])

    with pytest.raises(ValueError, match="Expected 2 visibility flags"):
        publisher.set_hidden_environments([True])

    events.clear()
    scene.close()
    assert events == [("destroy_paths", _BODY_PATHS), ("close_stage",)]
    assert scene._publishers == {}


@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires CUDA")
def test_device_poses_are_ordered_on_the_simulation_stream(monkeypatch):
    """CUDA writes are ordered on the stream that produced the poses, never on the "no sync" stream."""
    scene, events = _recording_scene(monkeypatch)
    backend, transforms = _physics("cuda:0")
    scene.publisher(
        BodyPosePublisher,
        lambda: BodyPosePublisher(scene, SceneDataProvider(backend), Usd.Stage.CreateInMemory(), "cuda:0"),
    )
    _set_positions(backend, transforms, (1, 2, 3))

    scene.publish()

    assert events[-1][2]["cuda_stream"] == (wp.get_device("cuda:0").stream.cuda_stream or 1)


def test_stage_publishes_scaled_poses_and_hidden_environments_without_a_renderer():
    """Physics poses reach a populated stage across commits, keeping authored scale and hiding environments."""
    usd = Usd.Stage.CreateInMemory()
    for env in _ENV_PATHS:
        UsdGeom.Xform.Define(usd, env)
    for path in _BODY_PATHS:
        UsdGeom.Xform.Define(usd, path).AddScaleOp().Set(Gf.Vec3d(2, 3, 4))
    backend, transforms = _physics()

    with contextlib.closing(OvstageBackend(OvstageBackendCfg())) as scene:
        scene.clone_copies = [(path, []) for path in _BODY_PATHS]
        scene.clone_env_paths = _ENV_PATHS
        scene.populate(usd.ExportToString())
        publisher = scene.publisher(
            BodyPosePublisher, lambda: BodyPosePublisher(scene, SceneDataProvider(backend), usd, "cpu")
        )

        def read(path: str, ordinal: int) -> np.ndarray:
            with (
                scene.query([path]) as query,
                scene.stage.read_attributes(
                    query, [scene.paths.intern_token("omni:xform")], ovstage.OrdinalRange.latest(ordinal)
                ) as result,
            ):
                result.wait()
                group = result.fetch_next()
                with group:
                    return np.array(group.columns[0].array(0)).reshape(4, 4)

        for position in ((1, 2, 3), (4, 5, 6)):
            _set_positions(backend, transforms, position)
            scene.publish()
            ordinal = scene.commit()
            for path in _BODY_PATHS:
                matrix = read(path, ordinal)
                np.testing.assert_allclose(matrix[3, :3], position)
                np.testing.assert_allclose(matrix[:3, :3], np.diag([2, 3, 4]))

        publisher.set_hidden_environments([False, True])
        scene.publish()  # Physics has not stepped.
        ordinal = scene.commit()
        np.testing.assert_allclose(read(_BODY_PATHS[0], ordinal)[3, :3], (4, 5, 6))
        np.testing.assert_allclose(read(_BODY_PATHS[1], ordinal)[3, :3], (4, 5, 6 - 1.0e6))
