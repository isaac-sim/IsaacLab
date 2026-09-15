// Copyright (c) 2022-2026, The Isaac Lab Project Developers.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

const DISCONNECT = 'The self-hosted runner lost communication with the server.';
const WORKFLOW = '.github/workflows/build.yaml';

function eligibleRun(run, repository) {
  return run?.repository?.full_name === repository &&
    run.path === WORKFLOW && run.event === 'pull_request' &&
    run.status === 'completed' && run.conclusion === 'failure' &&
    run.run_attempt === 1 && Number.isSafeInteger(run.id);
}

// Only GitHub's job-level annotation plus an empty step list qualifies. A test
// can print the same message; its output is deliberately never inspected here.
function disconnectedBeforeSteps(job, annotations, run) {
  return job.run_id === run.id && job.run_attempt === run.run_attempt &&
    job.head_sha === run.head_sha && Number.isSafeInteger(job.id) &&
    job.status === 'completed' && job.conclusion === 'failure' &&
    job.labels?.includes('self-hosted') && job.labels.includes('gpu') &&
    Array.isArray(job.steps) && job.steps.length === 0 &&
    annotations.some(annotation => annotation.path === '.github' &&
      annotation.annotation_level === 'failure' &&
      (annotation.message === DISCONNECT || annotation.message?.startsWith(`${DISCONNECT} `)));
}

async function recover({ github, context, core, dryRun = false }) {
  const scope = context.repo;
  const repository = `${scope.owner}/${scope.repo}`;
  const event = context.payload.workflow_run;
  const skip = reason => { core.info(`No automatic retry: ${reason}`); return { reason, retried: false }; };
  if (!eligibleRun(event, repository)) return skip('event is outside the first-attempt PR failure policy');

  const getRun = async () => (await github.rest.actions.getWorkflowRun({ ...scope, run_id: event.id })).data;
  const run = await getRun();
  if (!eligibleRun(run, repository) || run.head_sha !== event.head_sha) return skip('run changed since the event');
  const workflow = (await github.rest.actions.getWorkflow({ ...scope, workflow_id: 'build.yaml' })).data;
  if (run.workflow_id !== workflow.id) return skip('workflow identity does not match build.yaml');

  // workflow_run.pull_requests can be empty for fork PRs. Resolve by head and
  // verify the repository identity as well as the SHA, not the branch name alone.
  if (!run.head_repository?.owner?.login || !run.head_branch) return skip('missing source repository');
  const pulls = await github.paginate(github.rest.pulls.list, {
    ...scope, state: 'open', head: `${run.head_repository.owner.login}:${run.head_branch}`, per_page: 100,
  });
  const matches = pulls.filter(pr => pr.head?.repo?.id === run.head_repository.id &&
    pr.head.sha === run.head_sha && pr.base?.repo?.full_name === repository);
  if (matches.length !== 1) return skip('no unique open PR at the tested revision');
  const pr = matches[0];

  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRunAttempt, {
    ...scope, run_id: run.id, attempt_number: run.run_attempt, per_page: 100,
  });
  if (!jobs.length || jobs.some(job => job.status !== 'completed' ||
    !['success', 'skipped', 'neutral', 'failure'].includes(job.conclusion))) {
    return skip('incomplete, cancelled, timed-out, or unknown job outcome');
  }
  const failed = jobs.filter(job => job.conclusion === 'failure');
  if (!failed.length) return skip('no failed jobs');
  for (const job of failed) {
    // Construct API parameters from numeric IDs; never follow URLs in payloads.
    if (!Number.isSafeInteger(job.id)) return skip('invalid job identity');
    const prefix = `https://api.github.com/repos/${repository}/check-runs/`;
    if (typeof job.check_run_url !== 'string' || !job.check_run_url.startsWith(prefix)) {
      return skip('missing or foreign check identity');
    }
    const checkId = job.check_run_url.slice(prefix.length);
    if (!/^\d+$/.test(checkId) || !Number.isSafeInteger(Number(checkId))) return skip('invalid check identity');
    const annotations = await github.paginate(github.rest.checks.listAnnotations, {
      ...scope, check_run_id: Number(checkId), per_page: 100,
    });
    if (!disconnectedBeforeSteps(job, annotations, run)) return skip('a failed job is not a pre-step disconnect');
  }

  // A newer label-triggered run supersedes this run even on the same SHA.
  // This REST query is server-side bounded to this workflow and branch.
  const newer = await github.paginate(github.rest.actions.listWorkflowRuns, {
    ...scope, workflow_id: run.workflow_id, branch: run.head_branch,
    event: 'pull_request', created: `>=${run.created_at}`, per_page: 100,
  });
  if (newer.some(other => other.id !== run.id && other.head_repository?.id === run.head_repository.id &&
    (other.created_at > run.created_at || other.id > run.id))) return skip('a newer run supersedes this run');

  // Recheck volatile state immediately before the single write. A concurrent
  // human rerun or push must not spend another attempt on an obsolete result.
  const latest = await getRun();
  const currentPr = (await github.rest.pulls.get({ ...scope, pull_number: pr.number })).data;
  if (!eligibleRun(latest, repository) || latest.head_sha !== run.head_sha ||
    currentPr.state !== 'open' || currentPr.head?.sha !== run.head_sha ||
    currentPr.head?.repo?.id !== run.head_repository.id) return skip('run or PR changed during inspection');

  const ids = failed.map(job => job.id);
  core.info(`${dryRun ? 'Would retry' : 'Requesting retry for'} run ${run.id}, attempt 1, jobs ${ids.join(', ')}`);
  if (dryRun) return { retried: false, eligible: true, jobs: ids };
  // One request retries all failed jobs and their dependents. The all-failures
  // gate above prevents silently retrying assertions alongside a disconnect.
  // Do not retry this POST on transport errors: its outcome may be unknown.
  await github.rest.actions.reRunWorkflowFailedJobs({ ...scope, run_id: run.id });
  await core.summary.addHeading('Runner disconnect recovery')
    .addRaw(`Requested one retry of run ${run.id} after pre-step disconnects in jobs ${ids.join(', ')}.\n`)
    .addRaw('The original attempt remains available in the run history. Attempt 2 is never automatically retried.\n')
    .write();
  return { retried: true, jobs: ids };
}

module.exports = { recover, eligibleRun, disconnectedBeforeSteps };
