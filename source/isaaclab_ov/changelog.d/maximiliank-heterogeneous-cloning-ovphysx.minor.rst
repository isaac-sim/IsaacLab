Added
^^^^^

* Added plan-driven OvPhysX cloning for heterogeneous rigid-body and articulation geometry variants. GPU cloning imported complete original worlds and assigned one native environment ID per copied world; CPU imported full USD copies with collision grouping.
* Validated variant body/joint connectivity, articulation enablement, tendon layouts, and effective D6 rotational axes before cloning.

Fixed
^^^^^

* Preserved numeric environment order in tensor bindings, including non-cloned multi-instance assets, so indexed state reads and writes address the intended instances.
* Paired each cloned contact sensor with its own environment's filters and shared filters, and rejected missing filters that would silently remove sensor rows.
* Preserved all active source variants and independently authored assets during physics export, including custom environment templates and scaled asset roots.
* Removed destination placeholders and empty exported environment roots that caused quadratic native binding lookup costs.
* Registered Newton USD schemas before OVStage parsing. This affected all authored physics assets, including mimic-joint constraints, not only cloned assets.
* Deferred scene-data tensor bindings until visualization requests them, avoiding unnecessary headless-training startup cost.
