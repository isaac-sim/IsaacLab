Changed
^^^^^^^

* Changed boolean-mask scalar assignments to use :meth:`torch.Tensor.masked_fill_`
  in the Shadow Hand feature extractor, the Cartpole camera environments, and the
  drone navigation observations to avoid the ``nonzero``-based advanced-indexing
  path and improve performance.
