from copy import deepcopy
from typing import List, Any, Callable, Tuple

import numpy as np

from gpulink import NVContext
from gpulink.stoppable_thread import StoppableThread
from gpulink.types import GPURecording, RecType


class Recorder(StoppableThread):
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

    def _check_create_key(self, create_key):
        if create_key != Recorder.__create_key:
            raise RuntimeError("Recorder has to be instantiated using one of the factory methods!")

    @classmethod
    def create_memory_recorder(cls, ctx: NVContext, gpus: List[int]):
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

    def get_recording(self) -> GPURecording:
        recording = {
            "type": self._rec_type,
            "gpus": self._gpus,
            "gpu_names": [self._ctx.gpu_names[idx] for idx in self._gpus],
            "timestamps": [np.array(deepcopy(t)) for t in self._timestamps],
            "data": [np.array(deepcopy(t)) for t in self._data]
        }
        if self._rec_type:
            recording["max_values"] = [info.total for info in self._ctx.get_memory_info(self._gpus)]
        else:
            raise ValueError(f"Unsupported recording type: {self._rec_type}")

        return GPURecording(**recording)
