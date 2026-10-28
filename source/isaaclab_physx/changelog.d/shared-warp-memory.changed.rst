* Reused native Warp indexed views and copy/fill for selected asset writes, normalizing strided/int64 selectors once and keeping simulator row-ID handling in the backend.
* Used tiled concatenation for packed contact and frame-transformer poses, reusing recorded CUDA commands until sensor buffers are recreated.
