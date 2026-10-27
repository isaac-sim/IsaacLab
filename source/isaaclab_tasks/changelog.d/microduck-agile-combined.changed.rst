* Changed MicroDuck flat and rough walking to use the AGILE combined recipe by default, including with backlash.
  Custom tasks can retain the previous recipe by overriding the tracking rewards and action-rate curriculum
  and setting ``rewards.pose.params["standing_only"] = False``.
