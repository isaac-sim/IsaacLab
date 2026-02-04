from .measurements import Measurement, MetadataBase
from dataclasses import dataclass, field
from collections.abc import Sequence
from pathlib import Path


@dataclass
class MeasurementData:
    """Container for recorder measurements, metadata, and artifacts.

    Args:
        measurements: Recorded measurement values. Defaults to an empty list.
        metadata: Recorded metadata values. Defaults to an empty list.
        artefacts: Artifact tuples of (path, label). Defaults to an empty list.
    """

    measurements: Sequence[Measurement] = field(default_factory=lambda: [])
    metadata: Sequence[MetadataBase] = field(default_factory=lambda: [])
    artefacts: Sequence[tuple[Path, str]] = field(default_factory=lambda: [])  # (path, artefact-label)

class MeasurementDataRecorder:
    """Base class for recording metrics, metadata, and file-based artifacts.

    There are two common recorder styles: instantaneous measurements taken at
    a point in time, and sampling-based measurements gathered over a period.
    """

    def __init__(self):
        pass

    def update(self) -> None:
        pass

    def get_data(self) -> MeasurementData:
        """Return measurements, metadata, and artifacts collected so far.

        Returns:
            The collected measurement data container.

        Example:

        .. code-block:: python

            data = recorder.get_data()
        """
        return MeasurementData()