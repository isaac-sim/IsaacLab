Added
^^^^^

* Added PPISP camera QA demos for Kit and kit-less OVRTX workflows that render
  USD-authored PPISP scenes and save baseline, PPISP, and difference image
  grids.

Fixed
^^^^^

* Fixed :func:`~isaaclab_ppisp.auto_camera_ppisp_cfg` discovery so generated
  ``RenderProduct`` prims without PPISP do not mask later camera-bound PPISP
  render products.
