* Articulation, rigid-object and rigid-object-collection resets honour ``env_mask`` for actuators and external
  wrenches, so the per-step masked scene reset leaves unselected environments unchanged.
* **Breaking:** The backend event terms select environments with ``env_mask`` where the backend writes support masks.
