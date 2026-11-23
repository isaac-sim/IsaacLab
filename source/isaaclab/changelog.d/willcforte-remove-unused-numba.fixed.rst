* Removed the unused ``numba`` dependency, which could not be imported: the ``numpy>=2`` uv override
  discarded numba 0.64.0's ``numpy<2.5`` requirement, so NumPy 2.5 was installed alongside it.
