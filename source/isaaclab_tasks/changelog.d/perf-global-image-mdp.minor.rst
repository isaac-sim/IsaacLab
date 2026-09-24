Changed
^^^^^^^

* Changed the Kuka Allegro Lift camera observations to use the shared
  :func:`~isaaclab.envs.mdp.observations.image` term with ``stationary=True``. The observation
  values are unchanged.

Removed
^^^^^^^

* **Breaking:** Removed ``isaaclab_tasks.core.lift.mdp.vision_camera``. Use
  :func:`~isaaclab.envs.mdp.observations.image` with ``data_type=None``, ``permute=True``, and
  ``stationary=True`` instead, which produces the same observations.
