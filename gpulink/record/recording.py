from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from tabulate import tabulate

from gpulink.consts import MB, SEC
from gpulink.devices.gpu import GpuSet
from gpulink.record.timeseries import TimeSeries
from gpulink.plotting.plot_options import PlotOptions


@dataclass
class Recording:
    """
    A container for storing a gpu recording
    """
    gpus: GpuSet  # The recorded Gpu devices
    timeseries: List[TimeSeries]  # The recorded time series data
    plot_options: Optional[PlotOptions] = None  # Optional Plot options

    def _create_data_table(self):
        table = [["GPU", "Name", f"{self.plot_options.y_axis_label} [{self.plot_options.y_axis_unit}]"]]
        for gpu, timeseries in zip(self.gpus, self.timeseries):
            data = timeseries.data
            table.append(
                [gpu.id,
                 gpu.name,
                 f"minimum: {np.min(data) / MB}\nmaximum: {np.max(data) / MB}"
                 ]
            )
        return tabulate(table, tablefmt='fancy_grid')

    def _get_duration(self):
        timestamps = [t.timestamps for t in self.timeseries]
        min_time = min([np.min(t) for t in timestamps])
        max_time = min([np.max(t) for t in timestamps])
        return (max_time - min_time) / SEC

    def __str__(self):
        data_table = self._create_data_table()
        duration = self._get_duration()
        sampling_rate = self.timeseries[0].data.size / duration
        return f"{data_table}\n" \
               f"Recording duration:\t\t{duration:.3f} [s]\n" \
               f"Recording sampling rate:\t{sampling_rate:.3f} [Hz]"