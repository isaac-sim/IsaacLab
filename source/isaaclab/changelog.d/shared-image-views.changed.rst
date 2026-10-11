* Added shared ``ImageViewCfg`` selections for scene cameras and perspective renders.
  Windows reused one composed device image and retained native sensor dimensions.
  Configurations used ``NewtonGLVisualizerCfg(view=view, window=WindowCfg(...))`` or the RTX equivalent;
  legacy camera configurations remained supported. The existing simulation resource registry owned views.
* Bound camera references during simulation initialization and removed unused environment camera helpers.
  Kept image composition and channel selection in the shared image utilities.
* **Breaking:** Declared physics backend identifiers on their managers, preserving backend dispatch and viewer
  capabilities for solver subclasses with different names. Custom physics managers now declare
  ``backend_name`` explicitly; solver subclasses inherit it from their backend manager.
