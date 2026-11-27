* Fixed rigid object collections checking destination prims before replication when variants are authored
  under the prototype root. Construction now resolves authored sources through the existing clone-plan query.
  Collections also use Newton's body layout independently of shape layout, allowing members with different
  collider counts while preserving batched body state access.
