# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation adapters for processing named sensor buffers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.processing import ProcessingBufferPipe, ProcessingBufferStepCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class processed_sensor(ManagerTermBase):
    """Process named sensor render outputs with the shared buffer pipe.

    The sensor must expose ``request_render_inputs`` and ``render_outputs``, as
    :class:`~isaaclab.sensors.Camera` does. Operations define the supported buffer layouts and
    value semantics; outputs need not be visual images.

    Dependencies are requested before simulation initialization through
    :meth:`~isaaclab.managers.ObservationManager.prepare_scene`. They remain private to the
    renderer buffer bundle and need not appear in ``CameraCfg.data_types``. Each term owns
    its own operation state and output storage. The pipe runs on every observation evaluation;
    returned tensors are snapshots, so later evaluations cannot overwrite retained history.

    Use this term for multiple render inputs or changed buffer dimensions. To process other
    sensor data or the result of earlier observation modifiers, use
    :class:`~isaaclab.utils.modifiers.ProcessingBufferPipeModifier` on an observation tensor.
    To read a camera's already processed public output, use the existing image observation terms.
    """

    @classmethod
    def prepare_scene(cls, cfg: ObservationTermCfg, env: ManagerBasedEnv) -> None:
        sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("camera"))
        output_name = cfg.params.get("output_name", "rgb")
        requirements = ProcessingBufferPipe.required_source_names(cfg.params["pipeline"], [output_name])
        env.scene.sensors[sensor_cfg.name].request_render_inputs(requirements)

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self._output_name = cfg.params["output_name"]
        buffers = self._sensor.render_outputs
        device = wp.get_device(self.device)
        stream = wp.stream_from_torch(torch.cuda.current_stream(str(device))) if device.is_cuda else None
        with wp.ScopedStream(stream, sync_enter=True, sync_exit=True):
            self._pipe = ProcessingBufferPipe(buffers, cfg.params["pipeline"])
        if self._output_name not in self._pipe.outputs:
            self._pipe.close()
            raise ValueError(f"Buffer processing does not provide output {self._output_name!r}.")

    def __call__(
        self,
        env: ManagerBasedEnv,
        pipeline: Sequence[ProcessingBufferStepCfg],
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        output_name: str = "rgb",
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a snapshot of the selected operation output.

        The observation manager supplies ``out`` to receive the snapshot without another clone.
        Direct callers may omit it to allocate a new snapshot.
        """
        return self._pipe.as_tensor(self._sensor.render_outputs, output_name=self._output_name, out=out)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._pipe.reset(env_ids)

    def close(self) -> None:
        self._pipe.close()
