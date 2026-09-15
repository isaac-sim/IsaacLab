Added
^^^^^

* Added the opt-in training ``--export_deployment_usd`` flag to save ``deployment.usda`` and export timing/memory metrics in the run log directory. Global rank zero constructed the fixed single-environment task in an isolated process before event creation.
* Preserved export failure diagnostics and nonzero exit status when shutting down Isaac Sim.
