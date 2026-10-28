* Reused native Warp indexed views and copy/fill for selected asset writes and root-column reads; shared kernels handle boolean masks and fused state layouts.
* Used tiled concatenation for packed contact and frame-transformer poses, reusing recorded CUDA commands until sensor buffers are recreated.
