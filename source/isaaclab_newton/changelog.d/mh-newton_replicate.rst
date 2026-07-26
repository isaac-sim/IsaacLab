Changed
^^^^^^^

* Improved Newton scene cloning to use the batched ``replicate`` fast path for
  homogeneous environments, including those that register sensor sites (frame
  transformers, ray casters, IMUs) and per-world env-root sites, instead of
  building each world in a per-environment loop. This lowers environment-creation
  time for single-source, all-identical scenes. Scenes with multiple clone
  sources or MPM/deformable objects are unchanged and continue to use the
  per-world path.
