Fixed
^^^^^

* Followed USD dependencies through both local layers and remote assets before adding references, resolving nested remote references in working copies without editing authored layers or raw downloads. Preserved renderer-provided MDL identifiers and self-contained USDZ packages. Completed local trees skipped repeated traversal until their files changed or disappeared.
