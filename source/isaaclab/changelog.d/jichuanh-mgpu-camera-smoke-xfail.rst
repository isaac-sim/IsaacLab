Added
^^^^^

* Added multi-GPU training smoke tests that launch real four-rank training runs, covering each
  physics and renderer stack both with the host's default device order and with the devices
  exposed out of order, which is the case that catches renderer device-selection defects.
