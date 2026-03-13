# CLAUDE.md - Claude Code Configuration for Isaac Lab

## Primary Reference

All project guidelines, coding conventions, testing workflows, and contribution rules
are in **[AGENTS.md](AGENTS.md)**. Read it first — it is the authoritative guide for
any agent (human or AI) working on this repository.

## Claude-Specific Notes

### Skill Files

Claude Code skills for Isaac Lab are in `.claude/skills/isaaclab/`. These provide
agent-specific guidance for navigating the codebase, understanding the architecture,
and common workflows. See [.claude/skills/isaaclab/SKILL.md](.claude/skills/isaaclab/SKILL.md).

### Running Commands

Always use `./isaaclab.sh -p` to run Python scripts (never bare `python3`).
See AGENTS.md for the full CLI reference.

### Key Documentation Paths

| Topic | Path |
|-------|------|
| Installation | `docs/source/setup/installation/` |
| Tutorials | `docs/source/tutorials/` |
| Migration (Lab 3.0) | `docs/source/migration/migrating_to_isaaclab_3-0.rst` |
| API reference | `docs/source/api/` |
| Architecture | `docs/source/refs/reference_architecture/` |
