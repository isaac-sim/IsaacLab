Fixed
^^^^^

* Fixed :attr:`~isaaclab_teleop.XrCfg.near_plane` having no effect in XR sessions running
  under the ``ar`` or ``vr`` device profile (e.g. CloudXR): the profile-specific carb
  settings take precedence over the generic ``/persistent/xr/render/nearPlane`` and are
  now written as well (both in the XR anchor manager and the deprecated ``OpenXRDevice``).
