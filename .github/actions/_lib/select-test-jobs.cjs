// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Test-only PRs need their owning suites, not every integration/benchmark lane.
// Runtime, fixtures, unknown paths and incomplete API results retain full coverage.
module.exports = async ({github, context, core}) => {
  let selected = ['all'];
  if (context.eventName === 'pull_request') {
    try {
      const files = await github.paginate(github.rest.pulls.listFiles, {
        ...context.repo, pull_number: context.payload.pull_request.number, per_page: 100,
      });
      const owners = {
        isaaclab: ['core'],
        isaaclab_tasks: ['tasks', 'rendering', 'video', 'contrib'],
        isaaclab_assets: ['assets'],
        isaaclab_contrib: ['assets', 'contrib'],
        isaaclab_experimental: ['assets'],
        isaaclab_mimic: ['assets', 'curobo'],
        isaaclab_teleop: ['assets'],
        isaaclab_newton: ['newton'],
        isaaclab_physx: ['physx'],
        isaaclab_ov: ['ov'],
        isaaclab_rl: ['rl'],
        isaaclab_visualizers: ['visualizers'],
        _isaaclab_testing: ['rendering', 'visualizers'],
      };
      const lanes = new Set();
      // GitHub's PR files endpoint returns at most 3000 files.
      let complete = files.length > 0 && files.length < 3000;
      for (const file of files) {
        for (const path of [file.filename, file.previous_filename].filter(Boolean)) {
          const fragment = path.match(/^source\/([^/]+)\/changelog\.d\/[^/]+\.skip$/);
          if (fragment && owners[fragment[1]]) continue;
          const match = path.match(/^source\/([^/]+)\/test\/(?:[^/]+\/)*(test_[^/]+\.py)$/);
          if (!match || !owners[match[1]]) {
            complete = false;
            continue;
          }
          for (const lane of owners[match[1]]) lanes.add(lane);
          if (match[2] === 'test_standalone_scripts.py') lanes.add('demos');
          if (match[2].startsWith('test_curobo_') || match[2] === 'test_pink_ik.py') lanes.add('curobo');
        }
      }
      if (complete && lanes.size > 0) selected = [...lanes].sort();
    } catch (error) {
      core.warning(`Could not select test suites; running all suites: ${error.message}`);
    }
  }
  core.setOutput('lanes', JSON.stringify(selected));
  core.info(`Docker test lanes: ${selected.join(', ')}`);
};
