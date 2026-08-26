Changed
^^^^^^^

* Changed the homogeneous Newton cloning path to let replication name each cloned entity for
  the environment it lands in, instead of rewriting every replicated label afterwards. The
  prototype's labels are rebased once -- a few hundred entries -- and
  :meth:`~newton.ModelBuilder.replicate` is given the per-env roots, replacing a pass over
  every label in every world that cost 215 ms on ``Isaac-Velocity-Flat-G1`` at 4096
  environments. The labels are identical either way; a prototype whose labels a per-world
  prefix cannot spell keeps the previous path.
