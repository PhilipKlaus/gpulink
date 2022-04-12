from dataclasses import dataclass
from enum import Enum
from math import floor
from typing import List, Any, Union

import numpy as np
from tabulate import tabulate

from gpulink.consts import MB, SEC


class RecType(Enum):
    MEMORY_USED = "Memory"


_REC_TYPE_UNITS = {
    RecType.MEMORY_USED: "MB"
}


###########################################################################

@dataclass
class GPUQueryResult:
    """
    Represents the base class for a single GPU query result
    """
    timestamp: int
    gpu_idx: int
    gpu_name: str


@dataclass
class GPUQuerySingleResult(GPUQueryResult):
    value: Union[int, float, str]


@dataclass
class GPUMemInfo(GPUQueryResult):
    """
    Stores the result of a nvmlDeviceGetMemoryInfo query.
    """
    total: int
    used: int
    free: int


############################################################################

@dataclass
class GPURecording:
    """
    A container for storing a gpu recording
    """
    type: RecType  # The recording type
    gpus: List[int]  # List of GPU ids
    gpu_names: List[str]  # List of GPU names
    timestamps: List[np.ndarray]  # List of numpy arrays of recording timestamps
    data: List[np.ndarray]  # List of numpy arrays of recording data
    max_values: List[Any]  # List of maximum values of the actual recording data (per GPU)

    @property
    def duration(self):
        """
        Calculates the recording duration in Seconds.
        :return: The recording duration in seconds
        """
        return (self.timestamps[-1][-1] - self.timestamps[0][0]) / SEC

    @property
    def sampling_rate(self):
        """
        Calculates the average sampling rate (nvml calls per second).
        :return: The average sampling rate.
        """
        amount_records = (len(self.gpus) * self.timestamps[0].shape[0])
        return floor(amount_records / self.duration)

    def _create_metadata_table(self):
        return tabulate(
            [
                ["Record duration [s]", "Frame rate [Hz]"],
                [self.duration, self.sampling_rate]
            ],
            tablefmt='fancy_grid')

    def _create_data_table(self):
        table = [["GPU", "Name", f"{self.type.value} [{_REC_TYPE_UNITS.get(self.type)}]"]]
        for idx, name in zip(self.gpus, self.gpu_names):
            table.append([idx, name, f"minimum: {np.min(self.data) / MB}\nmaximum: {np.max(self.data) / MB}"])
        return tabulate(table, tablefmt='fancy_grid')

    def __str__(self):
        metadata_table = self._create_metadata_table()
        data_table = self._create_data_table()
        return f"{metadata_table}\n{data_table}"


##############################################################################

class TemperatureThreshold(Enum):
    TEMPERATURE_THRESHOLD_SHUTDOWN = 0
    TEMPERATURE_THRESHOLD_SLOWDOWN = 1
    TEMPERATURE_THRESHOLD_MEM_MAX = 2
    TEMPERATURE_THRESHOLD_GPU_MAX = 3
    TEMPERATURE_THRESHOLD_ACOUSTIC_MIN = 4
    TEMPERATURE_THRESHOLD_ACOUSTIC_CURR = 5
    TEMPERATURE_THRESHOLD_ACOUSTIC_MAX = 6
    TEMPERATURE_THRESHOLD_COUNT = 7


class ClockId(Enum):
    CLOCK_ID_CURRENT = 0
    CLOCK_ID_APP_CLOCK_TARGET = 1
    CLOCK_ID_APP_CLOCK_DEFAULT = 2
    CLOCK_ID_CUSTOMER_BOOST_MAX = 3


class ClockType(Enum):
    CLOCK_GRAPHICS = 0
    CLOCK_SM = 1
    CLOCK_MEM = 2
    CLOCK_VIDEO = 3


class TemperatureSensorType(Enum):
    GPU = 0
