Added
^^^^^

* Added multi-GPU troubleshooting documentation describing the NCCL hang that can occur when
  ``CUDA_VISIBLE_DEVICES`` selects a subset of a node's GPUs, including the
  ``NCCL_P2P_DISABLE=1`` workaround and an explanation of why it is not needed when all GPUs
  are visible.
