Changelog
---------

0.2.0 (2026-06-02)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :mod:`isaaclab_ppisp` package: a renderer-backend-agnostic post-render PPISP (Physically Plausible Image Signal Processing) pipeline that converts HDR scene-linear color to LDR RGBA at the end of a render tick.
* Added :class:`~isaaclab_ppisp.PpispPipeline`, :class:`~isaaclab_ppisp.PpispCfg`, the PPISP Warp kernel (:func:`~isaaclab_ppisp.apply_ppisp_to_rgba`), and the USD shader discovery helpers (:func:`~isaaclab_ppisp.auto_camera_ppisp_cfg`, :func:`~isaaclab_ppisp.auto_any_ppisp_cfg`).
* Backend renderers (:class:`~isaaclab_physx.renderers.IsaacRtxRenderer`, :class:`~isaaclab_ov.renderers.OVRTXRenderer`, :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`) compose :class:`~isaaclab_ppisp.PpispPipeline` internally when :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg` is set.
* Added PPISP camera QA demos for Kit and kit-less OVRTX workflows that render
  USD-authored PPISP scenes and save baseline, PPISP, and difference image
  grids.

Fixed
^^^^^

* Fixed :func:`~isaaclab_ppisp.auto_camera_ppisp_cfg` discovery so generated
  ``RenderProduct`` prims without PPISP do not mask later camera-bound PPISP
  render products.


0.1.0 (2026-05-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of ``isaaclab_ppisp`` providing the post-processing image-signal pipeline used by camera renderers.
