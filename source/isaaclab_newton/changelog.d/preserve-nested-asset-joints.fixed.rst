* Preserve joints and bodies beneath nested asset declarations in Newton clones without importing an identical child copy twice.
* Reject independently sourced nested Newton clone declarations that would overlap a parent's imported bodies.
* Preserve deformable particles and visual bindings when a cloned parent contains a declared deformable child.
* Preserve nested cloth across parent-only and child-only world compositions, including its configured world position.
* Rejected shared Newton roots that contain cloned environments; declare individual global asset paths instead.
