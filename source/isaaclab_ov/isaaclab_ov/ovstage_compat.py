# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVStage population and clone compatibility without loading the native runtime."""

from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from packaging.version import Version

if TYPE_CHECKING:
    import ovstage

try:
    OVSTAGE_VERSION = Version(version("ovstage"))
except PackageNotFoundError:
    OVSTAGE_VERSION = None

# The 0.4 release line needs CPU transform propagation before cloning, including development builds.
OVSTAGE_REQUIRES_CPU_HIERARCHY_BEFORE_CLONE = OVSTAGE_VERSION is not None and OVSTAGE_VERSION.release[:2] == (0, 4)


def populate_stage(
    stage: "ovstage.Stage", usda: str, *, ordinal: int, domains: "ovstage.PopulationDomain", physics: bool = False
) -> None:
    """Populate rendering and optional physics data across the OVStage 0.4 API transition.

    Older builds expose a physics domain bit; newer 0.4 builds use population descriptions.
    Detect the capability because both APIs shipped under the same release version.

    Description-based population selects physics properties through the OvPhysX SDK
    alongside the requested domains. Custom USD properties outside those selections
    are not copied to the native stage; the source USD scene is unchanged.

    Args:
        stage: Native stage to populate.
        usda: Exported USD scene.
        ordinal: Population write ordinal, sealed by the caller after cloning.
        domains: Native population domains, normally rendering.
        physics: Whether an OVPhysX consumer also needs physics data.
    """
    import ovstage

    if physics:
        physics_domain = getattr(ovstage.PopulationDomain, "PHYSICS", None)
        if physics_domain is None:
            ovstage.population.open_usd_from_string_with_desc(
                stage, usda, ordinal=ordinal, time=ovstage.population.DEFAULT_TIME,
                descs=(ovstage.population.Desc(domains=domains), _physics_population_desc()),
            )  # fmt: skip
            return
        domains |= physics_domain
    ovstage.population.open_usd_from_string(stage, usda, ordinal=ordinal, domains=domains)


@cache
def _physics_population_desc() -> "ovstage.population.Desc":
    from ovphysx.population import physics_population_desc

    return physics_population_desc()
