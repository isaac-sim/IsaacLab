:orphan:

.. _tutorial-deformable-gaussian-splats:


Rendering a deformable as Gaussian splats
=========================================

.. currentmodule:: isaaclab

Embed a Gaussian splat in a volume deformable so that the rendered splats follow the simulated body. This
tutorial holds a raspberry by its top nodes, shakes it, and saves the camera images. See :ref:`deformables`
for how splats are embedded and which renderers support them.

.. note::

   The tutorial uses Newton VBD. It renders with Newton Warp by default; ``--renderer ovrtx`` selects OVRTX
   and requires the ``ovrtx`` extra. ``--asset`` replaces the raspberry with your own splat asset.


The Code
~~~~~~~~

The tutorial corresponds to the ``run_deformable_gaussian_splats.py`` script in the ``scripts/tutorials/01_assets`` directory.

.. dropdown:: Code for run_deformable_gaussian_splats.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/01_assets/run_deformable_gaussian_splats.py
      :language: python
      :linenos:

.. code-block:: bash

   uv run python scripts/tutorials/01_assets/run_deformable_gaussian_splats.py --renderer newton_warp


The Code Explained
~~~~~~~~~~~~~~~~~~

Authoring the asset
-------------------

The raspberry is a USD file with a ``ParticleField3DGaussianSplat`` prim and a tetrahedral simulation mesh
under one deformable body. No visual mesh exists, so the splats are the only visual. Isaac Lab spawns it as
a :class:`assets.DeformableObjectCfg` and clones it into every environment.

.. literalinclude:: ../../../scripts/tutorials/01_assets/run_deformable_gaussian_splats.py
   :language: python
   :start-at: @configclass
   :end-before: def run_simulator(

The camera selects the renderer that draws the splats through its ``renderer_cfg``.

Moving the body
---------------

The splats are not driven directly. We constrain the top quarter of the nodes to a kinematic target that moves
side to side, using :meth:`assets.DeformableObject.write_nodal_kinematic_target_to_sim_index`. The solver moves
the remaining nodes, and the renderer moves every splat with them.

.. literalinclude:: ../../../scripts/tutorials/01_assets/run_deformable_gaussian_splats.py
   :language: python
   :start-at: # Constrain the top quarter
   :end-at: berry.update(sim_dt)

Reading the images
------------------

After each step, the camera's ``rgb`` output holds one image per environment. The script saves them every
quarter second to ``--output_dir``. Each environment simulates and renders its own body.
