Fixed
^^^^^

* Fixed ``HEADLESS=1`` combined with ``--visualizer kit`` aborting with
  ``AttributeError: module 'usdrt' has no attribute 'hierarchy'`` when the Newton backend set up its
  USD/Fabric sync. The headless experience now declares ``omni.hydra.usdrt_delegate``, which every
  other experience receives implicitly from the RTX renderer.
