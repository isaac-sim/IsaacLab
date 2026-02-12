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

# Type alias for metadata with data attribute (defined after classes below)
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

    data: str
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
        metadata_mapping = {str: StringMetadata, int: IntMetadata, float: FloatMetadata, dict: DictMetadata}
        for meas in m["metadata"]:
            if "data" in meas:
                metadata_type = metadata_mapping.get(type(meas["data"]))
                if metadata_type:
                    curr_meta = metadata_type(name=meas["name"], data=meas["data"])
                    metadata.append(curr_meta)
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
        curr_run = TestPhase(m["phase_name"])

        for meas in m["measurements"]:
            if "value" in meas:
                if isinstance(meas["value"], float):
                    curr_meas: Measurement = SingleMeasurement(
                        name=meas["name"], value=meas["value"], unit=meas["unit"]
                    )
                    curr_run.measurements.append(curr_meas)
                elif isinstance(meas["value"], dict):
                    curr_meas = DictMeasurement(name=meas["name"], value=meas["value"])
                    curr_run.measurements.append(curr_meas)
                elif isinstance(meas["value"], list):
                    curr_meas = ListMeasurement(name=meas["name"], value=meas["value"])
                    curr_run.measurements.append(curr_meas)
            elif "bvalue" in meas:
                curr_meas = BooleanMeasurement(name=meas["name"], bvalue=meas["bvalue"])
                curr_run.measurements.append(curr_meas)

            curr_run.metadata = TestPhase.metadata_from_dict(m["metadata"])
        return curr_run

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
        # Gather the separate metrics files for each test
        test_runs = []
        metric_files = os.listdir(json_folder_path)
        for f in metric_files:
            metric_path = os.path.join(json_folder_path, f)
            if os.path.isfile(metric_path):
                if f.startswith("metrics") and f.endswith(".json"):
                    with open(metric_path) as json_file:
                        try:
                            test_run_json_list = json.load(json_file)
                            for m in test_run_json_list:
                                run = cls.from_json(m)
                                test_runs.append(run)
                        except json.JSONDecodeError:
                            logger.error(
                                f'aggregate_json_files, problems parsing field {f} with content "{json_file.read()}"'
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
