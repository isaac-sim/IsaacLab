# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Isaac RTX (Replicator) Renderer."""

from typing import Literal

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass


@configclass
class IsaacRtxRendererCfg(RendererCfg):
    """Configuration for Isaac RTX renderer using Omniverse Replicator.

    Holds the Replicator/RTX-specific knobs (semantic segmentation, instance
    segmentation, semantic filtering, depth clipping) used by the RTX rendering
    pipeline.
    """

    renderer_type: str = "isaac_rtx"
    """Type identifier for Isaac RTX renderer."""

    semantic_filter: str | list[str] = "*:*"
    """A string or a list specifying a semantic filter predicate. Defaults to ``"*:*"``.

    If a string, it should be a disjunctive normal form of (semantic type, labels). For examples:

    * ``"typeA : labelA & !labelB | labelC , typeB: labelA ; typeC: labelE"``:
      All prims with semantic type "typeA" and label "labelA" but not "labelB" or with label "labelC".
      Also, all prims with semantic type "typeB" and label "labelA", or with semantic type "typeC" and label "labelE".
    * ``"typeA : * ; * : labelA"``: All prims with semantic type "typeA" or with label "labelA"

    If a list of strings, each string should be a semantic type. The segmentation for prims with
    semantics of the specified types will be retrieved. For example, if the list is ["class"], only
    the segmentation for prims with semantics of type "class" will be retrieved.

    .. seealso::

        For more information on the semantics filter, see the documentation on `Replicator Semantics Schema Editor`_.

    .. _Replicator Semantics Schema Editor: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html#semantics-filtering
    """

    colorize_semantic_segmentation: bool = True
    """Whether to colorize the semantic segmentation images. Defaults to True.

    If True, semantic segmentation is converted to an image where semantic IDs are mapped to colors
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """

    colorize_instance_id_segmentation: bool = True
    """Whether to colorize the instance ID segmentation images. Defaults to True.

    If True, instance id segmentation is converted to an image where instance IDs are mapped to colors.
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """

    colorize_instance_segmentation: bool = True
    """Whether to colorize the instance segmentation images. Defaults to True.

    If True, instance segmentation is converted to an image where instance IDs are mapped to colors.
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """

    semantic_segmentation_mapping: dict = {}
    """Dictionary mapping semantics to specific colours

    Eg.

    .. code-block:: python

        {
            "class:cube_1": (255, 36, 66, 255),
            "class:cube_2": (255, 184, 48, 255),
            "class:cube_3": (55, 255, 139, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (61, 178, 255, 255),
        }

    """

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
    """Clipping behavior for the camera for values exceed the maximum value. Defaults to "none".

    - ``"max"``: Values are clipped to the maximum value.
    - ``"zero"``: Values are clipped to zero.
    - ``"none"``: No clipping is applied. Values will be returned as ``inf``.
    """
