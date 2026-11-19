* Coupled native rigid-body and articulation-root pose writes with PhysX publication invalidation,
  preserving native tensor views and the existing deferred kinematics and Fabric refresh paths.
* Fixed same-step RTX frame reuse after tensor pose writes, including writes after a Kit visualizer frame.
