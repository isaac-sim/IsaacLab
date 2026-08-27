.. tab-set::

   .. tab-item:: Newton Warp

      .. figure:: ../../_static/overview/sensors/camera-renderer-newton.webp
         :align: center
         :width: 90%
         :alt: Six material spheres falling onto a table in Newton Warp RGB output.

         RGB animation

      .. grid:: 1 2 3 3
         :gutter: 2

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-newton-albedo.png
               :width: 100%
               :alt: Newton Warp albedo output for six material spheres.

               Albedo

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-newton-depth.png
               :width: 100%
               :alt: Color-mapped Newton Warp depth output for six material spheres.

               Depth

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-newton-normals.png
               :width: 100%
               :alt: Newton Warp surface-normal output for six material spheres.

               Normals

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-newton-semantic-segmentation.png
               :width: 100%
               :alt: Color-mapped Newton Warp semantic-segmentation output for six material spheres.

               Semantic segmentation

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-newton-instance-segmentation.png
               :width: 100%
               :alt: Color-mapped Newton Warp instance-segmentation output for six material spheres.

               Instance segmentation

   .. tab-item:: OVRTX

      .. figure:: ../../_static/overview/sensors/camera-renderer-ovrtx.webp
         :align: center
         :width: 90%
         :alt: Six material spheres falling onto a table in OVRTX RGB output.

         RGB animation

      .. grid:: 1 2 3 3
         :gutter: 2

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-ovrtx-albedo.png
               :width: 100%
               :alt: OVRTX albedo output for six material spheres.

               Albedo

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-ovrtx-depth.png
               :width: 100%
               :alt: Color-mapped OVRTX depth output for six material spheres.

               Depth

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-ovrtx-normals.png
               :width: 100%
               :alt: OVRTX surface-normal output for six material spheres.

               Normals

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-ovrtx-semantic-segmentation.png
               :width: 100%
               :alt: Color-mapped OVRTX semantic-segmentation output for six material spheres.

               Semantic segmentation

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-ovrtx-instance-segmentation.png
               :width: 100%
               :alt: Color-mapped OVRTX instance-segmentation output for six material spheres.

               Instance segmentation

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-ovrtx-motion-vectors.png
               :width: 100%
               :alt: Color-mapped OVRTX motion-vector output for six falling material spheres.

               Motion vectors

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-ovrtx-simple-shading-constant-diffuse.png
               :width: 100%
               :alt: OVRTX constant-diffuse simple-shading output for six material spheres.

               Constant diffuse shading

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-ovrtx-simple-shading-diffuse-mdl.png
               :width: 100%
               :alt: OVRTX diffuse-MDL simple-shading output for six material spheres.

               Diffuse MDL shading

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-ovrtx-simple-shading-full-mdl.png
               :width: 100%
               :alt: OVRTX full-MDL simple-shading output for six material spheres.

               Full MDL shading

   .. tab-item:: Isaac RTX

      .. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx.webp
         :align: center
         :width: 90%
         :alt: Six material spheres falling onto a table in Isaac RTX RGB output.

         RGB animation

      .. grid:: 1 2 3 3
         :gutter: 2

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-albedo.png
               :width: 100%
               :alt: Isaac RTX albedo output for six material spheres.

               Albedo

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-depth.png
               :width: 100%
               :alt: Color-mapped Isaac RTX depth output for six material spheres.

               Depth

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-normals.png
               :width: 100%
               :alt: Isaac RTX surface-normal output for six material spheres.

               Normals

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-semantic-segmentation.png
               :width: 100%
               :alt: Color-mapped Isaac RTX semantic-segmentation output for six material spheres.

               Semantic segmentation

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-instance-segmentation.png
               :width: 100%
               :alt: Color-mapped Isaac RTX instance-segmentation output for six material spheres.

               Instance segmentation

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-motion-vectors.png
               :width: 100%
               :alt: Color-mapped Isaac RTX motion-vector output for six falling material spheres.

               Motion vectors

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-simple-shading-constant-diffuse.png
               :width: 100%
               :alt: Isaac RTX constant-diffuse simple-shading output for six material spheres.

               Constant diffuse shading

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-simple-shading-diffuse-mdl.png
               :width: 100%
               :alt: Isaac RTX diffuse-MDL simple-shading output for six material spheres.

               Diffuse MDL shading

         .. grid-item::

            .. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-simple-shading-full-mdl.png
               :width: 100%
               :alt: Isaac RTX full-MDL simple-shading output for six material spheres.

               Full MDL shading

The editable scene and capture tools are available at
``tools/docs/media/renderer_gallery_scene.usda``,
``tools/docs/media/capture_renderer_gallery.py``, and
``tools/docs/media/generate_renderer_gallery.sh``. After reviewing the NVIDIA Omniverse EULA,
regenerate every output from the repository root with:

.. code-block:: bash

   OMNI_KIT_ACCEPT_EULA=Y tools/docs/media/generate_renderer_gallery.sh
