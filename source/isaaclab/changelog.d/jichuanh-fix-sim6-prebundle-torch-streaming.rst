Fixed
^^^^^

* Fixed a crash (``undefined symbol: ncclDevCommCreate``) when launching the Isaac Sim
  streaming app (e.g. ``isaac-sim.streaming.sh`` / ``runheadless.sh``) from an Isaac Lab
  install against Isaac Sim 6.0. Isaac Sim's deprecated ``omni.isaac.ml_archive`` prebundle
  ships its own PyTorch and NCCL while Isaac Lab installs a different pinned PyTorch; on
  launch paths that do not import Isaac Lab (which otherwise deprioritizes the prebundle on
  ``sys.path``), the two NCCL copies collide and the prebundled torch binds to the wrong
  one. The install step that repoints the prebundle to the active environment now uses
  overlayfs-safe filesystem operations, so it works inside the Docker image build (where it
  previously failed silently with ``EXDEV`` / ``EINVAL``) and fails loudly if a shadowing
  copy remains.
