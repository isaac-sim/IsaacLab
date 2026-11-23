* Added :attr:`~isaaclab.cloner.InclusionSet.match_tags` to declare variant compatibility using
  existing semantic tags, including multiple acceptable values. Matching groups stayed factorized
  until the scene's clone strategy selected concrete worlds, avoiding exhaustive combination
  expansion without inclusion callbacks, separate variant labels, or inclusion-owned sampling settings.
