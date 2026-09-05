Fixed
^^^^^

* Fixed ``--anim_recording_enabled`` silently running forever without saving an animation when the active
  physics backend is not PhysX. The OVD Recorder now raises a clear error at simulation startup naming the
  active backend and instructing the user to select the PhysX backend (for example, by appending
  ``physics=isaacsim_physx`` to the command line).
