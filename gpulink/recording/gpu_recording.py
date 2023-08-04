from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import numpy as np
from tabulate import tabulate

from gpulink.consts import SEC
from gpulink.devices.gpu import GpuSet
from gpulink.recording.timeseries import TimeSeries


class RecType(Enum):
    REC_TYPE_CLOCK_GRAPHICS = "Graphics Clock"
    REC_TYPE_CLOCK_SM = "Shared Memory Clock"
    REC_TYPE_CLOCK_MEM = "Memory Clock"
    REC_TYPE_CLOCK_VIDEO = "Video Clock"
    REC_TYPE_POWER_USAGE = "Power Usage"
    REC_TYPE_FAN_SPEED = "Fan Speed"
    REC_TYPE_TEMPERATURE = "Temperature"
    REC_TYPE_MEMORY = "Memory"


@dataclass
class Recording:
    """
    A container for storing a gpu recording
    """
    gpus: GpuSet  # The recorded Gpu devices
    timeseries: List[TimeSeries]  # The recorded time series data
    rtype: RecType
    name: str
    unit: str

    def _create_data_table(self):
        table = [["GPU", "Name", f"{self.name} ({self.rtype.value} [{self.unit}])"]]
        for gpu, timeseries in zip(self.gpus, self.timeseries):
            data = timeseries.data
            table.append(
                [gpu.id,
                 gpu.name,
                 f"minimum: {np.min(data)}\n"
                 f"maximum: {np.max(data)} "
                 ]
            )
        return tabulate(table, tablefmt='fancy_grid')

    def _get_duration(self):
        timestamps = [t.timestamps for t in self.timeseries]
        min_time = min([np.min(t) for t in timestamps])
        max_time = min([np.max(t) for t in timestamps])
        return (max_time - min_time) / SEC

    def convert(self, divider: Union[int, float], unit: str):
        for ts in self.timeseries:
            ts.apply_to_data(
                lambda data: data / divider
            )
        self.unit = unit

    def __str__(self):
        data_table = self._create_data_table()
        duration = f"{self._get_duration():.3f}"
        sampling_rate = f"{self.timeseries[0].data.size / max(self._get_duration(), sys.float_info.epsilon):.3f}"
        return f"{data_table}\n" \
               f"{'Duration:':25}{duration} [s]\n" \
               f"{'Sampling rate:':25}{sampling_rate} [Hz]"
