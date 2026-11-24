* Added :attr:`~isaaclab.cloner.InclusionSet.sample_by` to sample a semantic category before a variant
  within that category. Nested candidate groups stay factorized until clone planning selects concrete
  worlds, so large categories do not dominate small ones and category combinations need not be enumerated.
