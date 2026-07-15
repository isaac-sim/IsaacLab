Fixed
^^^^^

* Fixed PhysX CPU simulation initialization failing to create tensor views because
  no USD stage was attached.
* Fixed optional Isaac Sim extensions invalidating PhysX tensor views when they
  loaded Isaac Sim's simulation manager after Isaac Lab's physics manager.
* Fixed Kit viewport camera updates loading Isaac Sim's bundled Warp extension
  through ``isaacsim.core.rendering_manager``.
* Fixed the GUI application missing the Fabric extension required by the PhysX manager.
