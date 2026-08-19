Added
^^^^^

* Added support for the OVPhysX 0.6 ``warmup()`` API while retaining the
  released 0.5 ``warmup_gpu()`` path.
* Added support for the OVPhysX 0.6 ``destroy()`` lifecycle API while retaining
  the released 0.5 ``release()`` path.

Fixed
^^^^^

* Classified per-shape friction/restitution as CPU-resident. Reading it
  allocated on the simulation device, which OVPhysX 0.5 silently staged to the
  host on every call and OVPhysX 0.6 refuses outright.
