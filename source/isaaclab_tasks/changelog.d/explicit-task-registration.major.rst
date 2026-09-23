Changed
^^^^^^^

* **Breaking:** Importing ``isaaclab_tasks`` no longer registers every built-in
  environment with Gymnasium. Import ``isaaclab_tasks.registry`` before listing
  or creating built-in environments by name. Individual task modules can now be
  imported without loading unrelated task packages or registering their environments.
