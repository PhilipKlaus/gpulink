from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from tabulate import tabulate

from gpulink.consts import MB, SEC
from gpulink.devices.gpu import GpuSet
from gpulink.plotting.plot_options import PlotOptions
from gpulink.recording.timeseries import TimeSeries


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
        duration = f"{self._get_duration():.3f}"
        sampling_rate = f"{self.timeseries[0].data.size / self._get_duration():.3f}"
        return f"{data_table}\n" \
               f"Duration:\t{duration:11} [s]\n" \
               f"Sampling rate:\t{sampling_rate:11} [Hz]"
