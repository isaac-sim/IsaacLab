Fixed
^^^^^

* Followed USD dependencies through both local layers and remote assets before spawning, resolving nested remote references in working copies without editing authored layers or raw downloads. Completed local trees skipped repeated traversal until their files changed or disappeared.
