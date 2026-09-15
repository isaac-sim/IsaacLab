// Copyright (c) 2022-2026, The Isaac Lab Project Developers.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

const assert = require('node:assert/strict');
const { test } = require('node:test');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { runInNewContext } = require('node:vm');
const { recover, disconnectedBeforeSteps } = require('./recover-runner-disconnect.cjs');

// The incident shape: an assigned self-hosted job, no recorded steps, and a
// GitHub-generated .github annotation. IDs and repositories are synthetic.
function fixture() {
  const run = {
    id: 100, run_attempt: 1, workflow_id: 10, path: '.github/workflows/build.yaml',
    event: 'pull_request', status: 'completed', conclusion: 'failure',
    repository: { full_name: 'upstream/project' }, head_sha: 'a'.repeat(40),
    head_branch: 'topic', created_at: '2026-09-14T21:00:00Z', pull_requests: [],
    head_repository: { id: 20, owner: { login: 'contributor' } },
  };
  const job = {
    id: 200, run_id: run.id, run_attempt: 1, head_sha: run.head_sha,
    check_run_url: 'https://api.github.com/repos/upstream/project/check-runs/250',
    status: 'completed', conclusion: 'failure', labels: ['self-hosted', 'gpu'], steps: [],
  };
  const annotation = {
    path: '.github', annotation_level: 'failure',
    message: 'The self-hosted runner lost communication with the server. Verify the machine is running.',
  };
  const pr = {
    number: 42, state: 'open', head: { sha: run.head_sha, repo: { id: 20 } },
    base: { repo: { full_name: 'upstream/project' } },
  };
  const state = {
    run, latest: run, jobs: [job], annotations: [annotation], pulls: [pr], pr,
    newer: [run], writes: [], reads: [], messages: [], workflow: { id: 10 },
  };
  const method = name => name;
  let runReads = 0;
  const github = {
    rest: {
      actions: {
        getWorkflowRun: async args => {
          state.reads.push(['run', args]);
          return { data: ++runReads === 1 ? state.run : state.latest };
        },
        getWorkflow: async () => ({ data: state.workflow }),
        listJobsForWorkflowRunAttempt: method('jobs'),
        listWorkflowRuns: method('newer'),
        reRunWorkflowFailedJobs: async args => {
          state.writes.push(args);
          if (state.writeError) throw state.writeError;
        },
      },
      pulls: { list: method('pulls'), get: async () => ({ data: state.pr }) },
      checks: { listAnnotations: method('annotations') },
    },
    paginate: async (name, args) => {
      state.reads.push([name, args]);
      if (state.readError === name) throw new Error('API unavailable');
      return state[name];
    },
  };
  const summary = {
    addHeading() { return this; }, addRaw() { return this; }, async write() {},
  };
  const input = {
    github, context: { repo: { owner: 'upstream', repo: 'project' }, payload: { workflow_run: structuredClone(run) } },
    core: { info: message => state.messages.push(message), summary },
  };
  return { state, input, job, annotation };
}

test('fork PR with an empty event PR list gets one failed-jobs retry', async () => {
  const { state, input } = fixture();
  state.jobs.push({ ...state.jobs[0], id: 201 });
  assert.deepEqual(await recover(input), { retried: true, jobs: [200, 201] });
  assert.deepEqual(state.writes, [{ owner: 'upstream', repo: 'project', run_id: 100 }]);
  assert.deepEqual(state.reads.find(([name]) => name === 'jobs')[1], {
    owner: 'upstream', repo: 'project', run_id: 100, attempt_number: 1, per_page: 100,
  });
  assert.equal(state.reads.find(([name]) => name === 'pulls')[1].head, 'contributor:topic');
  assert.equal(state.reads.find(([name]) => name === 'annotations')[1].check_run_id, 250);
});

test('dry run exercises classification and revalidation without a POST', async () => {
  const { state, input } = fixture();
  assert.deepEqual(await recover({ ...input, dryRun: true }), { retried: false, eligible: true, jobs: [200] });
  assert.deepEqual(state.writes, []);
});

for (const [name, patch] of Object.entries({
  'second attempt': { run_attempt: 2 },
  'successful run': { conclusion: 'success' },
  'cancelled run': { conclusion: 'cancelled' },
  'running run': { status: 'in_progress' },
  'push event': { event: 'push' },
  'manual dispatch': { event: 'workflow_dispatch' },
  'another workflow': { path: '.github/workflows/other.yml' },
  'another repository': { repository: { full_name: 'fork/project' } },
  'missing attempt': { run_attempt: undefined },
})) {
  test(`does not retry ${name}`, async () => {
    const { state, input } = fixture();
    Object.assign(input.context.payload.workflow_run, patch);
    assert.equal((await recover(input)).retried, false);
    assert.deepEqual(state.reads, []);
    assert.deepEqual(state.writes, []);
  });
}

for (const [name, patch] of Object.entries({
  'recorded step': { steps: [{ name: 'Set up job', conclusion: 'success' }] },
  'absent step data': { steps: undefined },
  'GitHub-hosted runner': { labels: ['ubuntu-latest'] },
  'other pool': { labels: ['self-hosted', 'cpu'] },
  'other attempt': { run_attempt: 2 },
  'other run': { run_id: 999 },
  'other commit': { head_sha: 'b'.repeat(40) },
  'invalid ID': { id: '../200' },
  'missing check': { check_run_url: undefined },
  'foreign check': { check_run_url: 'https://api.github.com/repos/attacker/project/check-runs/250' },
  'malformed check': { check_run_url: 'https://api.github.com/repos/upstream/project/check-runs/250?x=1' },
  'timeout': { conclusion: 'timed_out' },
  'cancellation': { conclusion: 'cancelled' },
  'pending job': { status: 'queued' },
})) {
  test(`does not retry a job with ${name}`, async () => {
    const { state, input, job } = fixture();
    Object.assign(job, patch);
    assert.equal((await recover(input)).retried, false);
    assert.deepEqual(state.writes, []);
  });
}

test('a test failure alongside a disconnect prevents rerunning either', async () => {
  const { state, input, job } = fixture();
  state.jobs.push({ ...job, id: 201, steps: [{ name: 'pytest', conclusion: 'failure' }] });
  assert.equal((await recover(input)).retried, false);
  assert.deepEqual(state.writes, []);
});

test('failure on a later API page is also classified', async () => {
  const { state, input, job } = fixture();
  state.jobs = Array.from({ length: 100 }, (_, i) => ({ ...job, id: 300 + i, conclusion: 'success' }));
  state.jobs.push(job);
  assert.equal((await recover(input)).retried, true);
});

for (const [name, patch] of Object.entries({
  'ordinary error': { message: 'Process completed with exit code 1.' },
  'test output impersonating a disconnect': { path: 'test.py' },
  'warning': { annotation_level: 'warning' },
  'embedded signature': { message: 'AssertionError: The self-hosted runner lost communication with the server.' },
})) {
  test(`does not accept ${name} as a disconnect`, async () => {
    const { state, input, annotation } = fixture();
    Object.assign(annotation, patch);
    assert.equal((await recover(input)).retried, false);
    assert.deepEqual(state.writes, []);
  });
}

test('missing annotations do not imply a disconnect', async () => {
  const { state, input } = fixture();
  state.annotations = [];
  assert.equal((await recover(input)).retried, false);
  assert.deepEqual(state.writes, []);
});

test('no failures does not trigger a retry', async () => {
  const { state, input } = fixture();
  state.jobs[0].conclusion = 'success';
  assert.equal((await recover(input)).retried, false);
  assert.deepEqual(state.writes, []);
});

for (const name of ['closed', 'updated', 'ambiguous', 'different fork']) {
  test(`does not retry a ${name} PR`, async () => {
    const { state, input } = fixture();
    if (name === 'closed') state.pulls = [];
    if (name === 'updated') state.pulls[0].head.sha = 'b'.repeat(40);
    if (name === 'ambiguous') state.pulls.push(structuredClone(state.pulls[0]));
    if (name === 'different fork') state.pulls[0].head.repo.id = 21;
    assert.equal((await recover(input)).retried, false);
    assert.deepEqual(state.writes, []);
    assert.equal(state.reads.some(([method]) => method === 'jobs'), false);
  });
}

test('workflow ID must match independently resolved build.yaml', async () => {
  const { state, input } = fixture();
  state.workflow.id = 11;
  assert.equal((await recover(input)).retried, false);
  assert.deepEqual(state.writes, []);
});

test('a newer run on the same fork and branch suppresses the retry', async () => {
  const { state, input } = fixture();
  state.newer.push({ ...state.run, id: 101 });
  assert.equal((await recover(input)).retried, false);
  assert.deepEqual(state.writes, []);
});

test('another fork using the same branch name does not suppress the retry', async () => {
  const { state, input } = fixture();
  state.newer.push({ ...state.run, id: 101, head_repository: { id: 21 } });
  assert.equal((await recover(input)).retried, true);
});

for (const name of ['human rerun', 'new push', 'closed PR', 'replaced fork']) {
  test(`revalidates after inspection: ${name}`, async () => {
    const { state, input } = fixture();
    state.pr = structuredClone(state.pr);
    if (name === 'human rerun') state.latest = { ...state.run, run_attempt: 2, status: 'queued' };
    if (name === 'new push') state.pr.head.sha = 'b'.repeat(40);
    if (name === 'closed PR') state.pr.state = 'closed';
    if (name === 'replaced fork') state.pr.head.repo.id = 21;
    assert.equal((await recover(input)).retried, false);
    assert.deepEqual(state.writes, []);
  });
}

for (const name of ['jobs', 'annotations', 'pulls', 'newer']) {
  test(`API failure during ${name} inspection fails closed`, async () => {
    const { state, input } = fixture();
    state.readError = name;
    await assert.rejects(recover(input), /API unavailable/);
    assert.deepEqual(state.writes, []);
  });
}

test('ambiguous POST outcome is surfaced without a second write', async () => {
  const { state, input } = fixture();
  state.writeError = new Error('connection lost after request');
  await assert.rejects(recover(input), /connection lost/);
  assert.equal(state.writes.length, 1);
});

test('duplicate delivery after accepted rerun does not make another request', async () => {
  const { state, input } = fixture();
  await recover(input);
  state.latest = { ...state.run, run_attempt: 2, status: 'queued' };
  assert.equal((await recover(input)).retried, false);
  assert.equal(state.writes.length, 1);
});

test('the observed annotation also qualifies without explanatory suffix', () => {
  const { state, job, annotation } = fixture();
  annotation.message = 'The self-hosted runner lost communication with the server.';
  assert.equal(disconnectedBeforeSteps(job, [annotation], state.run), true);
});

test('workflow invokes the tested module from the trusted checkout', async () => {
  const yaml = readFileSync(resolve(__dirname, '../workflows/recover-runner-disconnect.yml'), 'utf8');
  assert.match(yaml, /ref: \$\{\{ github.sha \}\}/);
  assert.match(yaml, /persist-credentials: false/);
  assert.match(yaml, /retries: 0/);
  assert.doesNotMatch(yaml, /ref:.*workflow_run|download-artifact|actions\/cache/);
  const script = yaml.split('          script: |\n')[1];
  assert.ok(script);
  const { state, input } = fixture();
  const sandbox = {
    ...input,
    require: path => {
      assert.equal(path, './.github/scripts/recover-runner-disconnect.cjs');
      return { recover };
    },
  };
  await runInNewContext(`(async () => {${script}})()`, sandbox);
  assert.equal(state.writes.length, 1);
});
