# Changelog Fragment Examples

## Contents

- Patch fragment
- Minor fragment
- Skip fragment

## Patch Fragment

Use `source/isaaclab/changelog.d/<slug>.rst` for a bug fix:

```rst
Fixed
^^^^^

* Fixed contact sensor reset behavior when environments are partially reset.
```

## Minor Fragment

Use `source/isaaclab/changelog.d/<slug>.minor.rst` for a new public feature:

```rst
Added
^^^^^

* Added :class:`~isaaclab.sensors.ExampleSensor` for configurable example sensing.
```

## Skip Fragment

Use `source/isaaclab/changelog.d/<slug>.skip` when a `source/<package>/` change is test-only and has no user-facing release note.
