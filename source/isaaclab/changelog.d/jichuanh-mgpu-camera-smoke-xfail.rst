Added
^^^^^

* Added multi-GPU training smoke tests that launch real two-rank runs and select their GPU pair
  by interconnect class, so cross-socket rendering is exercised instead of whichever pair happens
  to be ``cuda:0,cuda:1``. The cross-socket camera case is marked ``xfail`` for NVBUG#6565122.

* Added :func:`~isaaclab.test.utils.gpu_pairs_by_topology` to classify GPU pairs from
  ``nvidia-smi topo -m`` as same-switch, cross-socket, or unknown.
