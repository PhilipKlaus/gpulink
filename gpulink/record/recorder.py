from typing import List, Callable, Tuple, Union, Optional

from gpulink import DeviceCtx
from gpulink.consts import MB
from gpulink.devices.gpu import GpuSet
from gpulink.plotting.plot_options import PlotOptions
from gpulink.devices.query import QueryResult
from gpulink.record.recording import Recording
from gpulink.threading.stoppable_thread import StoppableThread
from gpulink.record.timeseries import TimeSeries


class Recorder(StoppableThread):

    def __init__(
            self,
            cmd: Callable[[DeviceCtx], List[QueryResult]],
            res_filter: Callable[[QueryResult], Union[int, float, str]],
            ctx: DeviceCtx,
            gpus: List[int],
            plot_options: Optional[PlotOptions] = None
    ):
        super().__init__()
        self._cmd = cmd
        self._filter = res_filter
        self._ctx = ctx
        self._gpus = gpus
        self._plot_options = plot_options

        self._timeseries = [TimeSeries() for _ in gpus]

    def _get_record(self) -> Tuple[List, List[int]]:
        data = []
        timestamps = []
        for result in self._cmd(self._ctx):
            data.append(self._filter(result))
            timestamps.append(result.timestamp)
        return timestamps, data

    def _fetch_and_store(self):
        timestamps, data = self._get_record()
        for idx, record in enumerate(zip(timestamps, data)):
            self._timeseries[idx].add_record(record[0], record[1])

    def run(self):
        while not self.should_stop:
            self._fetch_and_store()

    def get_recording(self) -> Recording:
        return Recording(
            gpus=GpuSet([self._ctx.gpus[idx] for idx in self._gpus]),
            timeseries=self._timeseries,
            plot_options=self._plot_options
        )

    @classmethod
    def create_memory_recorder(cls, ctx: DeviceCtx, gpus: List[int], plot_options: Optional[PlotOptions] = None):
        default_options = PlotOptions(
            plot_name="Memory usage",
            y_axis_unit="MB",
            y_axis_label="Memory usage",
            y_axis_divider=MB,
            y_axis_range=(0, max([mem.total for mem in ctx.get_memory_info()]))
        )
        if plot_options:
            default_options.patch(plot_options)
        return Recorder(
            cmd=lambda c: c.get_memory_info(gpus),
            res_filter=lambda res: res.used,
            ctx=ctx,
            gpus=gpus,
            plot_options=default_options
        )
