from typing import List, Callable, Tuple, Union

from gpulink import DeviceCtx
from gpulink.stoppable_thread import StoppableThread
from gpulink.types import GPURecording, RecType, PlotInfo, GpuSet, TimeSeries, QueryResult


class Recorder(StoppableThread):

    def __init__(
            self,
            cmd: Callable[[DeviceCtx], List[QueryResult]],
            res_filter: Callable[[QueryResult], Union[int, float, str]],
            ctx: DeviceCtx,
            rec_type: RecType,
            gpus: List[int]
    ):
        super().__init__()
        self._cmd = cmd
        self._filter = res_filter
        self._ctx = ctx
        self._rec_type = rec_type
        self._gpus = gpus

        self._timeseries = [TimeSeries() for _ in gpus]

    # ToDo: Refactor away
    def _create_plot_info(self):
        if self._rec_type == RecType.MEMORY_USED:
            return PlotInfo(max_values=[info.total for info in self._ctx.get_memory_info(self._gpus)])
        else:
            return None

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
            plot_info=self._create_plot_info()
        )

    @classmethod
    def create_memory_recorder(cls, ctx: DeviceCtx, gpus: List[int]):
        return Recorder(
            cmd=lambda c: c.get_memory_info(gpus),
            res_filter=lambda res: res.used,
            ctx=ctx,
            rec_type=RecType.MEMORY_USED,
            gpus=gpus
        )
