Randomizing Rendered Appearance (Visual Domain Randomization)
=============================================================

Visual domain randomization (DR) varies a scene's *appearance* during training — color,
texture, lighting, materials — to make camera-based policies more robust to the sim-to-real
gap. This guide covers what visual DR is available out of the box and how it differs across
Isaac Lab's three rendering backends.

See :ref:`overview_renderers` for backend selection and
:doc:`/source/tutorials/03_envs/create_manager_rl_env` for event terms in general. This is the
*visual* counterpart to the *physics* DR in
:doc:`/source/tutorials/03_envs/create_direct_rl_env`.

Available out of the box
------------------------

Two Replicator-based event terms ship today and run on the **Isaac RTX renderer** (requires
Isaac Sim), applied via the event system at ``startup`` / ``reset`` / ``interval`` modes:

- :class:`~isaaclab.envs.mdp.events.randomize_visual_color` — randomizes diffuse color.
- :class:`~isaaclab.envs.mdp.events.randomize_visual_texture_material` — randomizes diffuse
  texture (from a preloaded list) and UV rotation.

A contrib dome-light term (intensity/color/texture) also exists, used by the visuomotor
stacking task and currently evaluation-gated.

.. note::

   The two material/texture terms require ``replicate_physics=False`` and randomize **all
   environments together** (no ``env_ids`` subsetting yet).

Backend support matrix
----------------------

"term shipped" = ready-to-use today; "runtime, no term" = the renderer reflects a runtime
write but no term wires it (an implementation target); "startup only" = fixed once the sim
runs; "unsupported" = no renderer API.

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Property
     - Isaac RTX
     - OVRTX (kit-less)
     - Newton Warp (kit-less)
   * - Diffuse color
     - runtime, **term shipped** (all envs)
     - runtime, no term (all envs)
     - runtime, no term (per-env)
   * - Texture (preloaded swap / UV)
     - runtime, **term shipped**
     - startup only
     - unsupported
   * - Material PBR scalars (roughness, metallic, …)
     - runtime, no term
     - runtime, no term
     - unsupported
   * - Light intensity / color
     - runtime (contrib term)
     - runtime, no term
     - startup only
   * - Dome-light HDR texture
     - runtime, expensive (contrib term)
     - startup only
     - startup only
   * - Per-environment addressing
     - not in terms today
     - not supported (instanced clones)
     - **native**
   * - Engine / pipeline / renderer type / tile layout
     - startup only
     - startup only
     - startup only

Cost (Isaac RTX)
----------------

Runtime DR is effectively free at ``startup`` / ``reset`` cadence; it only erodes throughput at
tight ``interval`` cadence and high env counts. A one-time startup cost de-instances visual
prims and creates per-prim materials (scales with prim count); per-call cost grows with env
count, and texture is several times costlier than color (it rebinds images). Prefer ``reset``
or a coarse ``interval`` for heavy properties.

.. note::

   RTX low-resolution output is not bit-reproducible
   (see :doc:`/source/features/reproducibility`); validate DR changes with a perceptual
   tolerance (SSIM / L2), not exact equality.

Extending DR (no term yet)
--------------------------

Newton per-env color is the cheapest to add (per-shape colors are read every frame, kit-less,
per-env); RTX PBR-scalar / light terms reuse the shipped pattern; OVRTX supports all-env
light/material writes kit-less (per-env needs de-instancing). When authoring a term, gate it on
the active renderer's capabilities and fail fast rather than silently no-op.

See Also
--------

- :ref:`overview_renderers`
- :doc:`/source/how-to/configure_rendering`
- :doc:`/source/how-to/cloning`
