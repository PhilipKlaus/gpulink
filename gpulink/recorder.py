from typing import List, Any, Callable, Tuple

from gpulink import DeviceCtx
from gpulink.factory import factory, make
from gpulink.stoppable_thread import StoppableThread
from gpulink.types import GPURecording, RecType, PlotInfo, GpuSet, TimeSeries


@factory
class Recorder(StoppableThread):
    """
    A recorder for GPU properties.
    """

    def __init__(self, ctx: DeviceCtx, rec_type: RecType, gpus: List[int], *args, **kwargs):
        super().__init__()
        self._ctx = ctx
        self._gpus = gpus
        self._rec_type = rec_type
        self._args = args
        self._kwargs = kwargs

        self._timestamps = [[] for _ in self._gpus]
        self._data = [[] for _ in self._gpus]

    @classmethod
    @make
    def create_memory_recorder(cls, key: object, ctx: DeviceCtx, gpus: List[int]):
        """
        Factory method to instantiate a Recorder.
        :param key: A factory key - must not be provided.
        :param ctx: A valid NVContext.
        :param gpus: The indices of the GPUs to be recorded.
        :return: An instance of Recorder.
        """
        return Recorder(key, ctx, RecType.MEMORY_USED, gpus)

    def _get_ctx_function(self) -> Tuple[Callable, Any]:
        tmp = {
            RecType.MEMORY_USED: {"function": self._ctx.get_memory_info, "rec_attr": "used"}
        }.get(self._rec_type, None)

        if not tmp:
            raise TypeError(f"Unsupported RecType: {self._rec_type}")
        return tmp["function"], tmp["rec_attr"]

    def run(self) -> None:

        function, attr = self._get_ctx_function()

        while not self.should_stop:
            res = function(self._gpus, *self._args, **self._kwargs)

            for i, entry in enumerate(res):
                t = entry.timestamp
                d = entry.__getattribute__(attr)
                self._data[i].append(d)
                self._timestamps[i].append(t)

    def _create_plot_info(self):
        if self._rec_type == RecType.MEMORY_USED:
            return PlotInfo(max_values=[info.total for info in self._ctx.get_memory_info(self._gpus)])
        else:
            return None

    def get_recording(self) -> GPURecording:
        """
        Get the actual recording data after the Recorder was stopped (using the stop() method)
        :return: Recording data as GPURecording.
        """
        return GPURecording(
            type=self._rec_type,
            gpus=GpuSet([self._ctx.gpus[idx] for idx in self._gpus]),
            timeseries=TimeSeries(self._timestamps, self._data),
            plot_info=self._create_plot_info()
        )
