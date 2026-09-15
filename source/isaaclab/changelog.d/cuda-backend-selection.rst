Added
^^^^^

* Added mutually exclusive ``cu128`` and ``cu130`` dependency groups for selecting PyTorch's CUDA build
  in uv source checkouts. Linux x86_64 and Windows retained CUDA 12.8 by default, and Linux
  aarch64 retained CUDA 13.0. Select CUDA 13.0 with ``--no-group cu128 --group cu130`` on
  both ``uv sync`` and ``uv run``.

Fixed
^^^^^

* Pinned the PyTorch stack in the published package dependencies so downstream projects no
  longer selected newer, untested builds when installing the Isaac Lab wheel.
