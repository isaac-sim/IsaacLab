* Kept PyTorch 2.12 with CUDA 13.0 while selecting the compatible NCCL 2.29.7 CUDA 12 build
  through uv to avoid collective initialization failures after RTX rendering starts.
* Propagated the NCCL exclusion to projects generated from a source checkout.
