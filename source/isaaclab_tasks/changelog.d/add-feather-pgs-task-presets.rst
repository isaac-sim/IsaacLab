Added
^^^^^

* Added the ``feather_pgs`` physics preset to supported task-family composition
  roots for the Newton FeatherPGS solver.

Changed
^^^^^^^

* Changed velocity tasks to share domain randomization, actuator conditioning,
  and PPO iteration budgets across physics solvers.
* Limited FeatherPGS presets to task families that do not require fixed tendons.

Fixed
^^^^^

* Fixed excessive FeatherPGS work in dexterous-hand tasks and unnecessary
  joint-limit row allocation in cabinet tasks.
* Fixed cabinet tasks to avoid repeating already stable FeatherPGS integration
  work while retaining the shared controller and collision cadence.
* Fixed non-finite FeatherPGS Reach simulations by enforcing the robots'
  authored joint-velocity limits across the task family.
* Fixed Allegro Hand FeatherPGS constraint truncation while retaining the
  shared simulation cadence.
