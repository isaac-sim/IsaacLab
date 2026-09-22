# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from pxr import UsdLux

import isaaclab.sim as sim_utils
from isaaclab.utils.string import to_camel_case

pytestmark = [pytest.mark.unit, pytest.mark.isaacsim_ci]

_COMMON = {"color": (0.1, 0.1, 0.1), "enable_color_temperature": True, "color_temperature": 5500.0, "intensity": 100.0}


@pytest.mark.parametrize(
    "cfg",
    [
        sim_utils.DiskLightCfg(radius=20.0, **_COMMON),
        sim_utils.DistantLightCfg(angle=20.0, **_COMMON),
        sim_utils.DomeLightCfg(texture_file="/path/to/sky.hdr", texture_format="latlong", **_COMMON),
        sim_utils.CylinderLightCfg(radius=20.0, length=2.0, treat_as_line=True, **_COMMON),
        sim_utils.SphereLightCfg(radius=20.0, treat_as_point=True, **_COMMON),
    ],
    ids=["disk", "distant", "dome", "cylinder", "sphere"],
)
def test_spawn_light(cfg):
    sim_utils.create_new_stage()
    prim = cfg.func("/World/light", cfg)

    assert prim.GetPath() == "/World/light"
    assert prim.GetTypeName() == cfg.prim_type
    # every config field maps onto a USD light attribute
    for attr_name, attr_value in cfg.to_dict().items():
        if attr_name in ("func", "prim_type", "visible", "semantic_tags", "copy_from_source", "spawn_path"):
            continue
        if attr_name == "texture_file":
            authored = UsdLux.DomeLight(prim).GetTextureFileAttr().Get().path
        elif attr_name == "texture_format":
            authored = UsdLux.DomeLight(prim).GetTextureFormatAttr().Get()
        elif attr_name == "visible_in_primary_ray":
            authored = prim.GetAttribute("visibleInPrimaryRay").Get()
        else:
            authored = prim.GetAttribute(f"inputs:{to_camel_case(attr_name, to='cC')}").Get()
        assert authored == attr_value, f"Failed for attribute: '{attr_name}'"

    with pytest.raises(ValueError, match="already exists"):
        cfg.func("/World/light", cfg)
