Changed
^^^^^^^

* Changed boolean-mask scalar assignments to use :meth:`torch.Tensor.masked_fill_`
  in the TacSL visuotactile renderer to avoid the ``nonzero``-based
  advanced-indexing path and improve performance.
