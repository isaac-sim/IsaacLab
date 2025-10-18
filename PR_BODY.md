Add multirotor/thruster actuator, multirotor asset and manager-based LMF2 drone task

Description

This PR introduces multirotor and thruster support and adds a manager-based example/task for the LMF2 drone. The change contains a new low-level thruster actuator model, a new `Multirotor` articulation asset class + configs, new thrust actions, and a manager-based drone task (LMF2) with MDP configs and RL agent configs.

Motivation and context
- Provides a reusable multirotor abstraction and a parameterized thruster actuator model so we can simulate multirotor vehicles (quad/hex/other).
- Adds a manager-based LMF2 drone task and configuration files to enable repro and training workflows for the LMF2 platform.
- Consolidates drone-specific code and prepares the repo for future control/sensor improvements.

Fixes: (no issue)

Type of change
- New feature (non-breaking addition of new functionality)
- Documentation update (added docs/comments where applicable)

Files changed (high-level summary)
- New/major files added:
  - source/isaaclab/isaaclab/actuators/thruster.py (new thruster actuator model)
  - source/isaaclab/isaaclab/assets/articulation/multirotor.py (new multirotor articulation)
  - source/isaaclab/isaaclab/assets/articulation/multirotor_cfg.py
  - source/isaaclab/isaaclab/assets/articulation/multirotor_data.py
  - source/isaaclab/isaaclab/envs/mdp/actions/thrust_actions.py
  - source/isaaclab_assets/isaaclab_assets/robots/lmf2.py and LMF2 URDF + asset files (lmf2.urdf, lmf2.zip, .asset_hash)
  - source/isaaclab_tasks/isaaclab_tasks/manager_based/drone_ntnu/* (new task code, commands, observations, rewards, state-based control configs and agent configs)
- Modified:
  - source/isaaclab/isaaclab/actuators/actuator_cfg.py (register thruster config)
  - source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py (register thrust actions)
  - small edits to various utils and types, and docs/make.bat
- Total diff (branch vs main when I checked): 33 files changed, ~2225 insertions, 65 deletions

Dependencies
- No new external top-level dependencies introduced. The branch adds assets (binary `.zip`) — ensure Git LFS is used if you want large assets tracked by LFS.
- The new drone task references standard repo-internal packages and Isaac Sim; no external pip packages required beyond the repo standard.

Checklist (status)
- [x] I have read and understood the contribution guidelines
- [x] I have run the `pre-commit` checks with `./isaaclab.sh --format`
- [ ] I have made corresponding changes to the documentation (recommend adding a short note in docs if needed)
- [ ] My changes generate no new warnings (some warnings may remain in tests — see below)
- [ ] I have added tests that prove my fix is effective or that my feature works (no new unit tests were added for thruster/multirotor in this branch; I recommend adding small unit tests around the thruster/compute API)
- [ ] I have updated the changelog and the corresponding version in the extension's `config/extension.toml` file (if this feature affects an extension)
- [ ] I have added my name to the `CONTRIBUTORS.md` or my name already exists there

Notes about the checks & tests I ran
- Formatting / pre-commit:
  - I ran: ./isaaclab.sh --format
  - Pre-commit initially reported several flake8 and YAML issues; I fixed those (imports, duplicated YAML key, a couple of comment styles) and re-ran format until pre-commit passed.
  - Current status: pre-commit hooks passed in the workspace.
- Tests:
  - I attempted to run the repo tests via: ./isaaclab.sh --test -q
  - Initially pytest was not available in the kit Python; I installed pytest into the kit Python used by the script.
  - Test run produced many test failures. The main reason is environment/packaging: tests expect the repo packages to be installed into the kit python environment (so `import isaaclab` works), and many tests need a working Isaac Sim environment or additional dependencies. Installing local packages into the kit python and configuring the Isaac Sim runtime would be required for a full, green test run.

Next steps
- If you want, I can push this branch and create the PR on GitHub and add the labels `asset` and `isaac-lab`.
- I can also install the local packages into the kit python and re-run the full test suite (takes longer) and update the PR with the test report.


