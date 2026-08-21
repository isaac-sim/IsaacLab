Changed
^^^^^^^

* **Breaking:** The Isaac RTX renderer now raises an error for albedo and simple-shading outputs on Isaac Sim
  versions before 6.0, rather than silently omitting them. Upgrade Isaac Sim or remove those data types.
