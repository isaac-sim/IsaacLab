# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation adapters for renderer-independent image processing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.images import RenderBufferPipe, RenderBufferStepCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class processed_image(ManagerTermBase):
    """Process named raw camera outputs with the shared render buffer pipe.

    Dependencies are requested before simulation initialization through
    :meth:`~isaaclab.managers.ObservationManager.prepare_scene`. They remain private to the
    renderer buffer bundle and need not appear in ``CameraCfg.data_types``. Each term owns
    its own operation state and output storage. The pipe runs on every observation evaluation;
    returned tensors are snapshots, so later evaluations cannot overwrite retained history.

    Use this term for multiple inputs or changed image dimensions. To process the result of
    earlier observation modifiers instead, use :class:`~isaaclab.utils.modifiers.RenderBufferPipeModifier`.
    To read a camera's already processed public output, use the existing image observation terms.
    """

    @classmethod
    def prepare_scene(cls, cfg: ObservationTermCfg, env: ManagerBasedEnv) -> None:
        sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("camera"))
        output_name = cfg.params.get("output_name", "rgb")
        requirements = RenderBufferPipe.required_source_names(cfg.params["pipeline"], [output_name])
        env.scene.sensors[sensor_cfg.name].request_render_inputs(requirements)

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._camera = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self._output_name = cfg.params["output_name"]
        self._pipe = RenderBufferPipe(self._camera.render_output_specs, cfg.params["pipeline"])
        if self._output_name not in self._pipe.output_specs:
            self._pipe.close()
            raise ValueError(f"Image processing does not provide output {self._output_name!r}.")

    def __call__(
        self,
        env: ManagerBasedEnv,
        pipeline: Sequence[RenderBufferStepCfg],
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        output_name: str = "rgb",
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a processed image in the layout, dtype and shape declared by its final operation.

        The observation manager supplies ``out`` to receive the snapshot without another clone.
        Direct callers may omit it to allocate a new snapshot.
        """
        return self._pipe.as_tensor(self._camera.render_outputs, output_name=self._output_name, out=out)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._pipe.reset(env_ids)

    def close(self) -> None:
        self._pipe.close()
