from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import List, Union, Iterator, Optional, Tuple

import numpy as np
from tabulate import tabulate

from gpulink.consts import MB


@dataclass
class Unit:
    name: str
    value: Union[int, float]


class RecType(Enum):
    """
    Recordable types of GPU properties.
    """
    MEMORY_USED = ("Memory used", Unit("MB", MB))


###########################################################################

@dataclass
class Gpu:
    """
    Represents the base class for a single GPU query result.
    """
    id: int
    name: str


class GpuSet:
    def __init__(self, gpus: List[Gpu]):
        self._gpus = gpus

    def __getitem__(self, key) -> Gpu:
        return self._gpus[key]

    def __len__(self) -> int:
        return len(self._gpus)

    def __eq__(self, other):
        return self.ids == other.ids and self.names == other.names

    @property
    def ids(self) -> List[int]:
        return [gpu.id for gpu in self._gpus]

    @property
    def names(self) -> List[str]:
        return [gpu.name for gpu in self._gpus]


@dataclass
class QueryResult:
    """
    Represents the base class for a single GPU query result.
    """
    timestamp: int
    gpu_idx: int
    gpu_name: str


@dataclass
class SimpleResult(QueryResult):
    """
    A GPUQueryResult containing only a single value.
    """
    value: Union[int, float, str]


@dataclass
class MemInfo(QueryResult):
    """
    Stores the result of a nvmlDeviceGetMemoryInfo query.
    """
    total: int
    used: int
    free: int


############################################################################

@dataclass
class TimeSeries:
    def __init__(self, timestamps: Optional[List[int]] = None, data: Optional[List[Union[int, float]]] = None):
        if data is None:
            data = []
        if timestamps is None:
            timestamps = []
        self._timestamps = timestamps
        self._data = data

    def add_record(self, timestamp, data):
        self._timestamps.append(timestamp)
        self._data.append(data)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array(self._timestamps)

    @property
    def data(self) -> np.ndarray:
        return np.array(self._data)


DATA = Union[int, float]


@dataclass
class PlotOptions:
    plot_name: Optional[str] = None
    y_axis_range: Optional[Tuple[DATA, DATA]] = None
    y_axis_label: Optional[str] = None
    y_axis_unit: Optional[str] = None
    y_axis_divider: Optional[DATA] = None

    def patch(self, other: PlotOptions):
        for field in fields(PlotOptions):
            value = other.__getattribute__(field.name)
            if value:
                self.__setattr__(field.name, value)


@dataclass
class PlotInfo:
    max_values: List[Union[int, float]]


@dataclass
class GPURecording:
    """
    A container for storing a gpu recording
    """
    rec_type: RecType  # The type of recording
    gpus: GpuSet  # The recorded Gpu devices
    timeseries: List[TimeSeries]  # The recorded time series data
    plot_options: Optional[PlotOptions] = None  # Optional Plot options

    def _create_metadata_table(self):
        return tabulate(
            [
                ["Record duration [s]", "Sampling rate [Hz]"],
                [self.timeseries.duration, self.timeseries.sampling_rate]
            ],
            tablefmt='fancy_grid')

    def _create_data_table(self):
        table = [["GPU", "Name", f"{self.rec_type.value[0]} [{self.rec_type.value[1].name}]"]]
        for gpu in self.gpus:
            table.append(
                [gpu.id,
                 gpu.name,
                 f"minimum: {np.min(self.timeseries.data) / MB}\nmaximum: {np.max(self.timeseries.data) / MB}"
                 ]
            )
        return tabulate(table, tablefmt='fancy_grid')

    def __str__(self):
        metadata_table = self._create_metadata_table()
        data_table = self._create_data_table()
        return f"{metadata_table}\n{data_table}"


##############################################################################

@dataclass
class SensorStatus:
    """
    A container for storing several sensor status.
    """
    gpus: GpuSet
    memory: List[MemInfo]
    temperature: List[SimpleResult]
    fan_speed: List[SimpleResult]
    clock: Iterator

    def __str__(self):
        header = ["GPU", "Memory [MB]", "Temp [°C]", "Fan speed [%]", "Clock [MHz]"]
        table = [header]
        for data in zip(self.gpus, self.memory, self.temperature, self.fan_speed, self.clock):
            table.append([
                f"GPU[{data[0].id}]",
                f"{int(data[1].used / MB)} / {int(data[1].total / MB)} ({(data[1].used / data[1].total) * 100:.1f}%)",
                f"{data[2].value}",
                f"{data[3].value}",
                f"Graph.: {data[4][0].value}\nMemory: {data[4][1].value}\n"
                f"SM: {data[4][2].value}\nVideo: {data[4][3].value} "
            ])
        return tabulate(table, headers='firstrow', tablefmt='fancy_grid')


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
