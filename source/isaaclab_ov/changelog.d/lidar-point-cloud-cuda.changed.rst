* Mapped the OVRTX lidar point cloud on CUDA and copied a dense tile cloud with one device kernel.
  The CPU map remains the fallback when the CUDA map is empty.
