Agent Skills
============

Isaac Lab agent skills are repository-owned instructions that help coding agents follow Isaac Lab workflows. They are markdown guidance assets, not runtime Python packages.

Skills live under the repository-level ``skills/`` directory:

.. code-block:: text

   skills/
   ├── developer/
   │   └── <skill-name>/
   │       ├── SKILL.md
   │       └── examples.md
   └── user/
       └── <skill-name>/
           ├── SKILL.md
           ├── evaluations.md
           ├── reference.md       # optional, recommended for longer guidance
           └── examples.md        # optional, recommended for concrete workflows

Developer skills
  Help contributors and maintainers follow Isaac Lab workflows such as PR preparation, changelog fragments, testing, and documentation rules.

User skills
  Ship with the repository as supported guidance for users building on top of Isaac Lab. Some user skills are intentionally router-only and may only need ``SKILL.md`` plus ``evaluations.md`` when the official docs and source examples already contain the details.

Skill files
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - File
     - Developer skills
     - User skills
     - Purpose
   * - ``SKILL.md``
     - Required
     - Required
     - Main agent-facing workflow and references.
   * - ``evaluations.md``
     - Optional
     - Required
     - Representative prompts, expected behavior, and pass/fail criteria.
   * - ``reference.md``
     - Optional
     - Recommended
     - Longer background that should not live in ``SKILL.md``.
   * - ``examples.md``
     - Recommended
     - Recommended
     - Concrete examples of the workflow.
   * - ``scripts/``
     - Optional
     - Optional
     - Deterministic helpers with explicit run-or-read instructions.

Discovery
---------

Codex and Claude discover the repository skills automatically through
project-native aliases:

.. code-block:: text

   .agents/skills/<frontmatter-name>  -> skills/<audience>/<slug>
   .claude/skills                     -> .agents/skills

Codex scans ``.agents/skills`` and Claude scans ``.claude/skills``. The aliases
use the frontmatter ``name`` so both agents expose the same stable identifier.
They resolve to the canonical directories under ``skills/``, keeping one
maintained copy of each skill and preserving repository-relative references.
Do not flatten-copy the skills into a global agent skill directory.

Agents without native skill discovery should use ``skills/README.md`` as the
catalog. Select a skill by matching the user's request against the
``description`` field in each ``SKILL.md`` frontmatter, then read only that
skill and its directly linked files. Directory slugs are canonical file
locations for humans and reviewers; the frontmatter ``name`` is the stable
identifier to use when one skill routes to another.

Skill contract
--------------

Every skill must include a ``SKILL.md`` file with frontmatter:

.. code-block:: yaml

   name: isaaclab-example-skill
   description: Performs a specific Isaac Lab workflow. Use when the user mentions the workflow or related trigger terms.
   audience: user
   status: stable
   owners:
     - isaaclab-maintainers

The required frontmatter fields are:

Frontmatter intentionally uses a small YAML subset: single-line scalar fields and the
``owners`` list. Do not use block scalars such as ``|`` or ``>`` in ``SKILL.md``
frontmatter; keep long details in the markdown body or a linked reference file.

``name``
  Unique lowercase identifier using letters, numbers, and hyphens. Prefer an ``isaaclab-`` prefix and avoid generic names such as ``helper``, ``utils``, or ``tools``. The identifier must match the native alias under ``.agents/skills/``.

``description``
  Third-person discovery text. Include what the skill does and when an agent should use it.

  Good example:

  .. code-block:: yaml

     description: Applies Isaac Lab coding style, API design, docstring, type-hint, lazy export, and contribution conventions. Use when writing or reviewing Isaac Lab Python code, public APIs, config classes, module exports, or documentation strings.

  Bad example:

  .. code-block:: yaml

     description: Helps with tasks.

``audience``
  Either ``developer`` or ``user``. This must match the directory under ``skills/``.

``status``
  One of ``experimental``, ``stable``, or ``deprecated``.

  Use ``experimental`` for new or incomplete guidance that still needs real usage. Promote a skill to ``stable`` only after it has maintained references, validation guidance, and review from the owning area. Use ``deprecated`` when a workflow is replaced or no longer recommended; deprecated skills must point to the replacement workflow or migration path.

``owners``
  One or more maintainer groups or owners responsible for the skill.

Required ``SKILL.md`` sections are:

* ``When To Use``
* ``Workflow``
* ``Validation``
* ``Maintenance``
* ``References``

User-facing skills must also include an ``evaluations.md`` file with at least three representative scenarios. Each scenario should include a sample prompt or task, expected behavior, and known failure modes or pass/fail criteria.

Authoring guidelines
--------------------

These guidelines are adapted from Anthropic's `Skill authoring best practices <https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices>`_ and tailored to Isaac Lab's documentation and contribution workflow.

Keep ``SKILL.md`` concise. Treat context as shared budget and assume agents already understand general programming concepts. Put only the task-critical workflow in ``SKILL.md``.

Use progressive disclosure. Link directly from ``SKILL.md`` to one-level files such as ``reference.md``, ``examples.md``, or ``evaluations.md``. Avoid chains where ``SKILL.md`` links to a file that links to the actual guidance.

Use a contents section for longer references. Reference files over 100 lines should start with ``# Contents`` or ``## Contents`` so agents can navigate them when reading partially.

Do not include credentials, private content copied from another system, generated logs, hardware-specific benchmark dumps, or large vendored documentation. If that information is useful, summarize the workflow and link to the maintained source instead.

Set the right degree of freedom:

* Use high-freedom guidance for reviews and design work.
* Use medium-freedom templates when Isaac Lab has a preferred pattern.
* Use low-freedom exact commands or scripts for fragile workflows such as validation.

Prefer one clear default before listing alternatives. If an escape hatch is needed, state when to use it.

Avoid time-sensitive wording. For migrations, use ``Current Workflow`` and ``Old Patterns`` sections instead of date-based instructions.

Use consistent Isaac Lab terminology throughout a skill, such as ``manager-based environments``, ``event terms``, ``articulation config``, and ``changelog fragment``.

Keeping skills synchronized
---------------------------

Skills should not become parallel documentation. The source of truth is the Isaac Lab documentation under ``docs/source/`` and the maintained source/examples under ``source/`` and ``scripts/``.

Each skill must include a ``Maintenance`` section that names the authoritative files to review when code changes. When a skill needs documentation-level content, update the official docs first and link to them from the skill.

Keep the link bidirectional. When a documentation page under ``docs/source/`` is the source of truth for a skill, add a ``.. seealso::`` admonition near the top of that page that names the skill and links back to it. This reminds documentation authors to update the associated skill in the same change, so the two never drift apart. Treat a missing back-link as a review issue whenever a skill's ``Maintenance`` section names a documentation page.

Use these rules during review:

* Reject large copied API tables or standalone install guides when official docs already cover the topic.
* Prefer links to docs, source files, and maintained examples over duplicated snippets.
* Keep only agent-specific value in the skill: routing, task sequencing, validation loops, search terms, and known decision points.
* If a skill uncovers a missing migration note or troubleshooting entry, add that note to ``docs/source/`` rather than expanding the skill.
* For simulation, asset, sensor, or randomization workflows, state whether the guidance is backend-neutral or specific to PhysX, Newton, or a renderer/visualizer backend. Link to the multi-backend docs and source implementations instead of copying backend tables into the skill.
* Run the skill validator after every skill change.

Validation workflow
-------------------

Validate skills locally with:

.. code-block:: bash

   uv run --no-project python tools/skills/cli.py check

The validator checks frontmatter, required sections, unique names, audience/path consistency, link validity, reference depth, portable paths, native Codex and Claude aliases, required user evaluations, minimum evaluation scenario counts, and per-scenario evaluation details such as sample queries, expected behavior, and known failure modes or pass/fail criteria.

Run the validator tests with:

.. code-block:: bash

   uv run --no-project --with pytest python -m pytest tools/skills/

The skills CI gate runs only when a pull request changes files under ``skills/``, ``tools/skills/``, or the skills workflow. This keeps unrelated Isaac Lab PRs from being blocked by skill validation while still checking validator changes.

Review policy
-------------

Developer skills require maintainer approval from the owning area because they affect contributor workflows.

User skills require maintained references, a validation path, documentation alignment, and at least three evaluation scenarios.

Reviewers should reject skills that duplicate general knowledge instead of encoding Isaac Lab-specific workflow. Deprecated skills must include migration guidance and a replacement skill when one exists.
