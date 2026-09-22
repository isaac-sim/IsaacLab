# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, cast

logger = logging.getLogger(__name__)

# Metadata records that carry a ``data`` attribute; the classes are defined below.
_MetadataWithData = Union["StringMetadata", "IntMetadata", "FloatMetadata", "DictMetadata"]


@dataclass
class Measurement:
    """Base measurement record.

    Args:
        name: Measurement name.
    """

    name: str


@dataclass
class SingleMeasurement(Measurement):
    """Single floating-point measurement.

    Args:
        name: Measurement name.
        value: Measurement value.
        unit: Unit string.
        type: Measurement type label. Defaults to "single".
    """

    value: float | int | str
    unit: str
    type: str = "single"


@dataclass
class StatisticalMeasurement(Measurement):
    """Statistical measurement.

    Args:
        name: Measurement name.
        mean: Mean value.
        std: Standard deviation value.
        n: Number of samples.
        unit: Unit string.
        type: Measurement type label. Defaults to "statistical".
    """

    mean: float
    std: float
    n: int
    unit: str
    type: str = "statistical"


@dataclass
class BooleanMeasurement(Measurement):
    """Boolean measurement.

    Args:
        name: Measurement name.
        bvalue: Measurement value.
        type: Measurement type label. Defaults to "boolean".
    """

    bvalue: bool
    type: str = "boolean"


@dataclass
class DictMeasurement(Measurement):
    """Dictionary measurement.

    Args:
        name: Measurement name.
        value: Measurement value.
        type: Measurement type label. Defaults to "dict".
    """

    value: dict
    type: str = "dict"


@dataclass
class ListMeasurement(Measurement):
    """List measurement.

    Args:
        name: Measurement name.
        value: Measurement value.
        type: Measurement type label. Defaults to "list".
    """

    value: list
    type: str = "list"

    def __repr__(self):
        """Return a compact string representation.

        Returns:
            String representation of the measurement.

        Example:

        .. code-block:: python

            repr_str = repr(ListMeasurement(name="samples", value=[1, 2, 3]))
        """
        return f"{self.__class__.__name__}(name={self.name!r}, length={len(self.value)})"


@dataclass
class MetadataBase:
    """Base metadata record.

    Args:
        name: Metadata name.
    """

    name: str


@dataclass
class StringMetadata(MetadataBase):
    """String metadata.

    Args:
        name: Metadata name.
        data: Metadata value.
        type: Metadata type label. Defaults to "string".
    """

    data: str | None
    type: str = "string"


@dataclass
class IntMetadata(MetadataBase):
    """Integer metadata.

    Args:
        name: Metadata name.
        data: Metadata value.
        type: Metadata type label. Defaults to "int".
    """

    data: int
    type: str = "int"


@dataclass
class FloatMetadata(MetadataBase):
    """Float metadata.

    Args:
        name: Metadata name.
        data: Metadata value.
        type: Metadata type label. Defaults to "float".
    """

    data: float
    type: str = "float"


@dataclass
class DictMetadata(MetadataBase):
    """Dictionary metadata.

    Args:
        name: Metadata name.
        data: Metadata value.
        type: Metadata type label. Defaults to "dict".
    """

    data: dict
    type: str = "dict"


_METADATA_TYPES: dict[type, type[_MetadataWithData]] = {
    str: StringMetadata,
    int: IntMetadata,
    float: FloatMetadata,
    dict: DictMetadata,
}


@dataclass
class TestPhase:
    """Represent a single test phase with associated metrics and metadata.

    Args:
        phase_name: Name of the phase.
        measurements: Measurements recorded for the phase. Defaults to an empty list.
        metadata: Metadata recorded for the phase. Defaults to an empty list.
    """

    phase_name: str
    measurements: list[Measurement] = field(default_factory=list)
    metadata: list[_MetadataWithData] = field(default_factory=list)

    def get_metadata_field(self, name: str, default: Any = KeyError) -> Any:
        """Get a metadata field's value.

        Args:
            name: Field name. Note that fields are named internally like 'Empty_Scene Stage DSSIM Status', however
                `name` is case-insensitive, and drops the stage name. In this eg it would be 'stage dssim status'.
            default: Default value to return when the field is missing.

        Returns:
            Metadata value, or default if provided.

        Raises:
            KeyError: If the field is not found and no default is provided.

        Example:

        .. code-block:: python

            status = phase.get_metadata_field("stage dssim status", default=None)
        """
        name = name.lower()
        for m in self.metadata:
            name2 = m.name.replace(self.phase_name, "").strip().lower()
            if name == name2:
                return cast(Any, m).data

        if default is KeyError:
            raise KeyError(name)
        return default

    @classmethod
    def metadata_from_dict(cls, m: dict) -> list[_MetadataWithData]:
        """Build metadata objects from a metadata dictionary.

        Args:
            m: Dictionary containing a "metadata" list.

        Returns:
            List of metadata objects.

        Example:

        .. code-block:: python

            metadata = TestPhase.metadata_from_dict({"metadata": [{"name": "gpu", "data": "A10"}]})
        """
        metadata: list[_MetadataWithData] = []
        for entry in m["metadata"]:
            metadata_type = _METADATA_TYPES.get(type(entry.get("data")))
            if metadata_type is not None:
                metadata.append(metadata_type(name=entry["name"], data=entry["data"]))
        return metadata

    @classmethod
    def from_json(cls, m: dict) -> "TestPhase":
        """Deserialize measurements and metadata from a JSON structure.

        Args:
            m: JSON-compatible dictionary containing phase data.

        Returns:
            Deserialized test phase object.

        Example:

        .. code-block:: python

            phase = TestPhase.from_json(phase_dict)
        """
        phase = TestPhase(m["phase_name"], metadata=cls.metadata_from_dict(m["metadata"]))
        for meas in m["measurements"]:
            if "value" in meas:
                value = meas["value"]
                if isinstance(value, float):
                    phase.measurements.append(SingleMeasurement(name=meas["name"], value=value, unit=meas["unit"]))
                elif isinstance(value, dict):
                    phase.measurements.append(DictMeasurement(name=meas["name"], value=value))
                elif isinstance(value, list):
                    phase.measurements.append(ListMeasurement(name=meas["name"], value=value))
            elif "bvalue" in meas:
                phase.measurements.append(BooleanMeasurement(name=meas["name"], bvalue=meas["bvalue"]))
        return phase

    @classmethod
    def aggregate_json_files(cls, json_folder_path: str | Path) -> list["TestPhase"]:
        """Aggregate test phases from JSON files in a folder.

        Args:
            json_folder_path: Folder containing metrics JSON files.

        Returns:
            List of aggregated test phases.

        Example:

        .. code-block:: python

            phases = TestPhase.aggregate_json_files("/tmp/metrics")
        """
        test_runs: list[TestPhase] = []
        for name in os.listdir(json_folder_path):
            metric_path = os.path.join(json_folder_path, name)
            if not (name.startswith("metrics") and name.endswith(".json") and os.path.isfile(metric_path)):
                continue
            with open(metric_path) as json_file:
                try:
                    test_runs.extend(cls.from_json(m) for m in json.load(json_file))
                except json.JSONDecodeError:
                    logger.error(
                        f'aggregate_json_files, problems parsing field {name} with content "{json_file.read()}"'
                    )
        return test_runs


class TestPhaseEncoder(json.JSONEncoder):
    """JSON encoder for test phases and measurement objects."""

    def default(self, o: object) -> dict:
        """Serialize objects by exposing their dictionary representation.

        Args:
            o: Object to serialize.

        Returns:
            Dictionary representation of the object.

        Example:

        .. code-block:: python

            json.dumps(phase, cls=TestPhaseEncoder)
        """
        return o.__dict__
