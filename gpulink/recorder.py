from copy import deepcopy
from typing import List, Any, Callable, Tuple

import numpy as np

from gpulink import NVContext
from gpulink.stoppable_thread import StoppableThread
from gpulink.types import GPURecording, RecType, PlotInfo


class Recorder(StoppableThread):
    """
    A recorder for GPU properties.
    """
    __create_key = object()

    def __init__(self, create_key, ctx: NVContext, rec_type: RecType, gpus: List[int], *args, **kwargs):
        super().__init__()
        self._check_create_key(create_key)
        self._ctx = ctx
        self._gpus = gpus
        self._rec_type = rec_type
        self._args = args
        self._kwargs = kwargs

        self._timestamps = [[] for _ in self._gpus]
        self._data = [[] for _ in self._gpus]

    @classmethod
    def _check_create_key(cls, create_key):
        if create_key != Recorder.__create_key:
            raise RuntimeError("Recorder has to be instantiated using one of the factory methods!")

    @classmethod
    def create_memory_recorder(cls, ctx: NVContext, gpus: List[int]):
        """
        Factory method to instantiate a Recorder.
        :param ctx: A valid NVContext.
        :param gpus: The indices of the GPUs to be recorded.
        :return: An instance of Recorder.
        """
        return Recorder(cls.__create_key, ctx, RecType.MEMORY_USED, gpus)

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
            gpus=self._gpus,
            gpu_names=[self._ctx.gpu_names[idx] for idx in self._gpus],
            timestamps=[np.array(deepcopy(t)) for t in self._timestamps],
            data=[np.array(deepcopy(t)) for t in self._data],
            plot_info=self._create_plot_info()
        )

