from typing import List, Callable, Tuple, Union, Optional

from gpulink import DeviceCtx
from gpulink.consts import MB
from gpulink.stoppable_thread import StoppableThread
from gpulink.types import GPURecording, RecType, GpuSet, TimeSeries, QueryResult, PlotOptions


class Recorder(StoppableThread):

    def __init__(
            self,
            cmd: Callable[[DeviceCtx], List[QueryResult]],
            res_filter: Callable[[QueryResult], Union[int, float, str]],
            ctx: DeviceCtx,
            rec_type: RecType,
            gpus: List[int],
            plot_options: Optional[PlotOptions] = None
    ):
        super().__init__()
        self._cmd = cmd
        self._filter = res_filter
        self._ctx = ctx
        self._rec_type = rec_type
        self._gpus = gpus
        self._plot_options = plot_options

        self._timeseries = [TimeSeries() for _ in gpus]

    def get_record(self) -> Tuple[List, List[int]]:
        data = []
        timestamps = []
        for result in self._cmd(self._ctx):
            data.append(self._filter(result))
            timestamps.append(result.timestamp)
        return timestamps, data

    def fetch_and_store(self):

        for idx, record in enumerate(self.get_record()):
            self._timeseries[idx].add_record(record[0], record[1])

    def run(self):
        while not self.should_stop:
            self.fetch_and_store()

    def get_recording(self) -> GPURecording:
        return GPURecording(
            rec_type=self._rec_type,
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
            rec_type=RecType.MEMORY_USED,
            gpus=gpus,
            plot_options=default_options
        )
