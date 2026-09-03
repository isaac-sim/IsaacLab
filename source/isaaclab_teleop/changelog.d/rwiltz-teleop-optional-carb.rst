Fixed
^^^^^

* Fixed ``from isaaclab_teleop import IsaacTeleopDevice`` raising ``ModuleNotFoundError: No module named 'carb'``
  on hosts without Isaac Sim installed. :mod:`~isaaclab_teleop.xr_anchor_manager` now imports ``carb`` with the
  same optional fallback it already used for ``omni.kit.xr.core``, so headless sessions that never start an XR
  runtime can import the device. The XR render and anchor settings are skipped when Kit is absent; behavior with
  Kit present is unchanged.
