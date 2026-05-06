Fixed
^^^^^

* Fixed three places where ``OvPhysxManager`` was misclassified as the
  PhysX backend by a substring/schema match:

  - :meth:`~isaaclab.sensors.SensorBase._register_callbacks` matched
    ``"physx" in physics_mgr_cls.__name__.lower()`` to gate the PhysX
    ``IsaacEvents.PRIM_DELETION`` import — the substring also matches
    ``"OvPhysxManager"``, so the ``isaaclab_physx`` import fired in
    kitless OVPhysX mode and raised
    :exc:`ModuleNotFoundError` because ``omni.physics.tensors`` is not
    loaded.  Switched to an exact ``physics_mgr_cls.__name__ ==
    "PhysxManager"`` match.
  - :meth:`~isaaclab.assets.AssetBase.set_debug_vis` had the same
    substring check guarding an ``import omni.kit.app`` call, which
    would fire for OVPhysX-backed assets and break under
    ``./scripts/run_ovphysx.sh``.  Switched to an exact
    ``"PhysxManager"`` match.
  - :meth:`~isaaclab.physics.SceneDataProvider._get_backend` used
    ``"physx" in manager_name`` to dispatch the backend factory; this
    silently routed ``OvPhysxManager`` to the PhysX scene-data
    provider.  Switched to exact ``"PhysxManager"`` /
    ``"NewtonManager"`` matches and an explicit ``ValueError`` for
    unknown managers.
* Made
  :attr:`~isaaclab.scene.InteractiveScene.physics_scene_path` accept a
  bare :class:`pxr.UsdPhysics.Scene` prim as a fallback when no prim
  with ``PhysxSceneAPI`` applied is on the stage.  Kitless OVPhysX
  does not load the ``omni.physx`` schema, so the auto-created scene
  prim only carries the stock USD type.  PhysX-backed flows continue
  to prefer the ``PhysxSceneAPI`` prim.
