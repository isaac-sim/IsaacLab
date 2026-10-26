* Extended ``TensorProcessor`` and ``TensorProcessorCfg`` to support tensors, ``ProxyArray`` values,
  and nested lists, tuples, and dictionaries through the shared ``TensorProcessorData`` type.
  Added optional named input/output metadata and lifecycle hooks while preserving single-tensor
  observation processing. Added manager resolution of nested scene entities and manager terms
  in processor parameters.
