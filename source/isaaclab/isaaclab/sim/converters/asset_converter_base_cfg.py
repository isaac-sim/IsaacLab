# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class AssetConverterBaseCfg:
    """The base configuration class for asset converters."""

    asset_path: str = MISSING
    """The absolute path to the asset file to convert into USD."""

    usd_dir: str | None = None
    """The output directory path to store the generated USD file. Defaults to None.

    If None, it is resolved as ``/tmp/IsaacLab/usd_{date}_{time}_{random}``, where
    the parameters in braces are runtime generated.
    """

    usd_file_name: str | None = None
    """The name of the generated usd file. Defaults to None.

    If None, it is resolved from the asset file name. For example, if the asset file
    name is ``"my_asset.urdf"``, then the generated USD file name is ``"my_asset.usd"``.

    If the providing file name does not end with ".usd" or ".usda", then the extension
    ".usd" is appended to the file name.
    """

    force_usd_conversion: bool = False
    """Force the conversion of the asset file to usd. Defaults to False.

    If True, then the USD file is always generated. It will overwrite the existing USD file if it exists.
    """

    make_instanceable: bool = True
    """Make the generated USD file instanceable. Defaults to True.

    Note:
        Instancing helps reduce the memory footprint of the asset when multiple copies of the asset are
        used in the scene. For more information, please check the USD documentation on
        `scene-graph instancing <https://openusd.org/dev/api/_usd__page__scenegraph_instancing.html>`_.
    """
