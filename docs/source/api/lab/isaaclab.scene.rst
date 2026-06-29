isaaclab.scene
==============

.. automodule:: isaaclab.scene

  .. rubric:: Classes

  .. autosummary::

    InteractiveScene
    InteractiveSceneCfg

  .. rubric:: Functions

  .. autosummary::

    scene_add

interactive Scene
-----------------

.. autoclass:: InteractiveScene
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: InteractiveSceneCfg
    :members:
    :exclude-members: __init__

Scene configuration composition
-------------------------------

Use :func:`scene_add` to combine spawned
:class:`~isaaclab.assets.AssetBaseCfg`-family fields at literal, one-segment
``/World/Leaf`` or ``{ENV_REGEX_NS}/Leaf`` roots. When both operands declare
global assets, their unordered asset sets must be native-equal, including prim
paths; field names do not affect the match. If only one operand declares a
global world, that world is carried into the result.
Each operand must contribute at least one spawned environment asset.
Preset-wrapped scene fields must be resolved before composition; registered task
configuration helpers perform this resolution automatically.

Use ``asset_skip`` when composition policy should omit selected spawned assets.
The predicate receives each :class:`~isaaclab.sim.SpawnerCfg` and returning
``True`` removes its owning scene asset from both matching and clone rows. Flat
terrain importers are lowered to :class:`~isaaclab.sim.GroundPlaneCfg` before
the predicate runs. For example, a caller can ignore every source light without
making lighting a built-in merge rule:

.. code-block:: python

    combined = scene_add(
        left,
        right,
        asset_skip=lambda asset: isinstance(asset, sim_utils.LightCfg),
    )

Environment-scoped field names define logical slots. A native-equal definition
in the same slot reuses its existing binding, while a different definition
receives a unique field name and only a colliding prim path is suffixed. A slot
cannot change between global and environment scope. Execution settings and the
clone strategy from the second operand are ignored, while its clone
combinations and weights are composed.

Sensors and spawnless assets are discarded; other unknown field types are
rejected. A flat :class:`~isaaclab.terrains.TerrainImporterCfg` becomes a
bounded, environment-local ground while preserving its diffuse color and
physics material. This is a clone-only scene conversion: it does not preserve
the :class:`~isaaclab.terrains.TerrainImporter` object, terrain-origin API,
debug-origin visualization, or terrain-dependent sensors. The result is not a
drop-in task environment scene. Generated and USD terrain, rigid-object
collections, selectors, malformed roots, and positive-weight clone rows that
become empty are rejected.

Environment-scoped :class:`~isaaclab.sim.GroundPlaneCfg` assets must use the
local collision group (``collision_group=0``) with collision filtering enabled,
and their visual size must not exceed the output environment spacing. The
physical plane remains infinite and relies on per-environment collision
isolation.

.. autofunction:: scene_add
