<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Physical Backend Documentation Reorganization

## Goal

Move the legacy physics-backend documentation out of `Overview > Core Concepts` and reorganize it by
content type. The result should give each subject one authoritative home, keep the top-level Concepts
section explanatory, and remove manually maintained material that duplicates installation, environment,
or generated API documentation.

This design covers the 16 requested pages at commit `ca5c4a8a9b5b5e783f503c58fa8df249adedb807`,
using their current `develop` versions as the editing baseline. It also covers the minimum companion
changes required for a coherent move: navigation, internal cross-references, the direct-access hub,
repository skills that declare these pages as sources of truth, and any files that link to moved pages.

## Documentation Model

The reorganization uses four reader-facing content categories:

1. **Concepts** explain terminology, mental models, architecture, and trade-offs.
2. **Setup** owns installation and environment creation.
3. **How-to** pages guide a reader toward a concrete outcome.
4. **API reference** owns exhaustive symbols, fields, defaults, and method signatures.

Repository-extension procedures belong in the existing **Developer guide** rather than in the user-facing
Concepts or How-to sections.

Every retained fact should have one primary owner. Other pages should summarize only enough context to
route the reader to that owner.

## Proposed Navigation

```text
Concepts
├── Backends and presets
├── Backend architecture
├── Physics backends
└── Solver differences

Setup
└── Installation

How-to
├── Prepare an asset for Newton
├── Tune MJWarp
├── Enable and tune Kamino
└── Access native physics APIs
    ├── PhysX
    ├── Newton
    └── OvPhysX

Developer guide
└── Add a physics backend

API reference
└── Generated backend configuration and native-access classes
```

`Backends and presets` remains the user entry point for selector semantics and task presets. `Physics
backends` becomes a separate, concise comparison page rather than expanding the existing entry point into
a long backend manual.

## Page Disposition

| Current page | Destination | Disposition |
| --- | --- | --- |
| `multi_backend_architecture.rst` | Concepts and Developer guide | Keep the factory, physics-manager, unified-interface, and design-principle explanations in `Backend architecture`. Move the procedural `Adding a New Physics Backend` section into its own developer guide. Remove selector and preset examples already owned by `Backends and presets`. |
| `physical-backends/index.rst` | Concepts | Rewrite as `Physics backends`. Keep backend-versus-solver terminology, stable decision criteria, and links to focused guidance. Replace the exhaustive capability matrix with a small comparison of runtime, maturity, primary use, and solver family. Remove the duplicate `PresetCfg` example. |
| `physx/index.rst` | Concepts | Merge its unique description into `Physics backends`; remove the standalone landing page. |
| `physx/installation.rst` | Setup | Delete the page and link to the canonical installation guide. Do not preserve its abbreviated installation procedure. |
| `physx/configuration.rst` | Concepts and API | Remove the standalone page. Preserve concise guidance about TGS versus PGS, static GPU buffers, and stabilization/contact-sensor interaction in `Solver differences`. Link to generated `PhysxCfg` documentation for fields and defaults. |
| `physx/supported-features.rst` | Concepts and existing component references | Delete the page. Do not carry forward the duplicated component inventory. |
| `newton/index.rst` | Concepts | Merge Newton's identity, maturity, and solver model into `Physics backends`; remove the standalone landing page and repeated navigation prose. |
| `newton/installation.rst` | Setup | Delete the page. The canonical installation guide already owns current uv, extras, Isaac Sim, platform, and verification instructions. |
| `newton/supported-features.rst` | Concepts, Environments, and API | Delete the manually maintained task, asset, sensor, and package inventories. Keep only stable maturity and limitation summaries in `Physics backends`. Route task availability to the environments page and symbol availability to generated API pages. |
| `newton/migrating-assets-from-physx-to-newton.rst` | How-to | Rewrite as `Prepare an asset for Newton`. Keep the ordered mechanical-model audit, backend-specific schema guidance, validation procedure, and failure-localization workflow. Remove duplicated MJWarp profiles and general actuator explanations; link to their owning pages. |
| `newton/mjwarp-solver.rst` | How-to | Rewrite as `Tune MJWarp`. Keep the diagnose-first workflow, capacity semantics, contact-path choice, measured tuning sequence, and warnings against compensating for invalid models. Link to generated API documentation for individual fields and defaults. |
| `newton/kamino-solver.rst` | How-to | Rewrite as `Enable and tune Kamino`. Keep PADMM-versus-DVI selection, preset integration, compatibility checks, and the tuning sequence. Remove exhaustive parameter tables already generated from the configuration classes. |
| `direct-api-access/physx.rst` | How-to | Move under `Access native physics APIs`. Retain lifecycle, ownership, synchronization, invalidation, and focused examples. |
| `direct-api-access/newton.rst` | How-to | Move under `Access native physics APIs`. Retain the live-data and selection mental model plus lifecycle responsibilities. |
| `direct-api-access/ovphysx.rst` | How-to | Move under `Access native physics APIs`. Retain raw binding and guarded-view ownership semantics. |
| `solver-comparison.rst` | Concepts | Rewrite as `Solver differences`. Keep behavioral explanations and porting implications. Move the procedural porting checklist to the Newton asset how-to and avoid copying field-level API reference. |

The existing `direct-api-access/index.rst` moves with its child pages and remains their comparison and
warning hub.

## Content Ownership

### Backends and presets

This page owns:

- the meanings of backend, solver, renderer, visualizer, and preset;
- selector syntax and resolution order;
- task-specific preset discovery;
- common preset naming conventions; and
- the author-facing `PresetCfg` example.

Other pages may show one minimal selector needed for their procedure but must not re-explain the preset
system.

### Backend architecture

This page owns:

- factory dispatch and lazy backend resolution;
- the role and lifecycle of `PhysicsManager`;
- portable asset, sensor, renderer, and scene-data interfaces; and
- the design boundary between unified Isaac Lab APIs and native engine APIs.

It must not contain a volatile component-support matrix or a step-by-step extension tutorial.

### Physics backends

This page owns:

- the identity and intended use of PhysX, Newton, and OvPhysX;
- the distinction between a backend and its solvers;
- a compact comparison based on stable decision factors; and
- maturity and limitation summaries with links to live sources.

It must not enumerate supported tasks, assets, sensors, or every configuration field.

### Solver differences

This page owns conceptual differences that explain why identical parameters do not produce identical
behavior. It may compare contact, friction, stabilization, coordinates, timesteps, convergence, and memory
models. Exact defaults remain in the API reference; procedural tuning remains in How-to.

### How-to pages

Each how-to starts with a concrete outcome, prerequisites, and an ordered procedure. Explanatory material
is retained only when it affects a decision in that procedure. Repeated command variants, task inventories,
and parameter catalogs are removed or replaced by links to maintained sources.

## Material to Remove

- Both backend-specific installation pages.
- Both backend-specific supported-feature pages.
- Standalone PhysX and Newton landing pages after their introductions are merged.
- The component implementation matrix from the architecture page.
- Repeated backend selector and `PresetCfg` examples outside `Backends and presets`.
- Hand-maintained task and asset support lists.
- The exhaustive Kamino configuration tables.
- Duplicated MJWarp starting profiles in the asset-migration guide.
- Copied field defaults and signatures already present in generated API documentation.
- Repeated maturity disclaimers and `See also` lists that only mirror the navigation tree.

Deletion does not mean discarding unique operational warnings. Those warnings move to the concept or
how-to page where the reader makes the corresponding decision.

## Scope Boundaries

The Newton tree contains newer pages not included in the requested list: VBD, MPM, cables, manager
abstractions, and Warp environments. This change does not rewrite or relocate those source files. Their
existing paths remain valid, but their navigation entries move into hidden toctrees under the closest new
How-to or Developer guide hub so that deleting the old Newton landing page does not orphan them. A
content-type audit and rewrite of those pages is a separate effort.

The change does not alter public APIs, runtime behavior, installation commands, task presets, or generated
API structure. Documentation claims that conflict with current code or canonical setup documentation are
corrected rather than preserved for historical continuity.

## Paths, Links, and Compatibility

- Preserve existing explicit Sphinx labels, such as `mjwarp-solver-tuning` and
  `newton-kamino-solver`, on the replacement pages when their meaning remains valid.
- Add stable labels to new concept and how-to entry points so future internal links need not depend on file
  paths.
- Update every in-repository `:doc:`, `:ref:`, literal include, skill reference, and navigation entry that
  points at a moved or deleted page.
- Remove old source files after internal references are migrated. The repository has no configured
  documentation-redirect mechanism, and the recent actuator move establishes deletion plus label
  preservation as the available precedent.
- Update skills that name an affected page as their source of truth in the same commit as the corresponding
  move.

## Validation

The implementation is complete when:

1. The top-level Concepts navigation exposes the four backend concept pages without the old backend tree
   appearing under `Overview > Core Concepts`.
2. Setup, How-to, Developer guide, and generated API pages each own only their intended content type.
3. Repository search finds no internal links to deleted paths and no duplicated task or asset inventories
   from the removed supported-feature pages.
4. The documentation builds with warnings treated as errors.
5. Repository-wide pre-commit hooks pass before commit.
6. Skills referencing moved source-of-truth pages validate and point to the new locations.

## Implementation Shape

Implement the reorganization in reviewable stages:

1. Establish the Concepts pages and remove duplicated conceptual material.
2. Move and rewrite the procedural How-to and Developer pages.
3. Delete superseded installation, feature-inventory, landing, and parameter-reference pages.
4. Update navigation, cross-references, and skills.
5. Run link searches, skill validation, the warnings-as-errors documentation build, and pre-commit.

These stages describe review boundaries, not separate long-lived compatibility states; the final branch must
contain one coherent navigation tree.
