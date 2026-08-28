Changed
^^^^^^^

* **Breaking:** Changed Franka Pour reset artifacts to store the robot asset path relative to the Isaac Lab
  asset root. Regenerate custom reset datasets with ``generate_franka_pour_reset_dataset.py`` before using
  them with the updated task.
