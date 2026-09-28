Changed
^^^^^^^

* Reduced Newton camera kernel launches by combining pose conversion with transform packing and
  rendering planar depth directly into camera output buffers, removing the intermediate ray-depth
  allocation and conversion pass. Camera output conventions and clipping behavior were preserved;
  no configuration changes are required.
