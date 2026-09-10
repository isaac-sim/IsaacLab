Changed
^^^^^^^

* Allowed the shared generalized-force ordering kernel to omit direction signs for backends that already returned
  forces in the public joint basis. Callers supplying direction signs retained their existing behavior.
