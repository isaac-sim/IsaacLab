* Preserved joints and bodies beneath nested asset declarations in Newton clones without importing an identical child copy twice.
* Rejected independently sourced nested Newton clone declarations that would overlap a parent's imported bodies.
* Preserved deformable particles and visual bindings when a cloned parent contained a declared deformable child.
* Preserved nested cloth across parent-only and child-only world compositions, including its configured world position.
* Rejected shared Newton roots that contain cloned environments; declare individual global asset paths instead.
