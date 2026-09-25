Fixed
^^^^^

* Published deformable geometry through SDP using clone-plan paths and native particle ranges,
  preserving nonconsecutive environment IDs, custom paths, and position writes between renders.
  Removed the internal Fabric-sync helper; rendering consumers now bind through SDP.
