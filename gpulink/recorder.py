from copy import deepcopy
from typing import List

import numpy as np

from gpulink import NVContext
from gpulink.stoppable_thread import StoppableThread
from gpulink.types import GPURecording, RecType


class Recorder(StoppableThread):
    def __init__(self, ctx: NVContext, rec_type: RecType, gpus: List[int]):
        super().__init__()
        self._ctx = ctx
        self._gpus = gpus
        self._rec_type = rec_type
        self._rec_type_map = {
            RecType.MEMORY_USED: {"function": ctx.get_memory_info, "rec_attr": "used", "kwargs": None}
        }
        self._timestamps = [[] for _ in gpus]
        self._data = [[] for _ in gpus]

    def set_arguments(self, **kwargs) -> None:
        self._rec_type_map[self._rec_type]["kwargs"] = kwargs

    def run(self) -> None:
        while not self.should_stop:
            function = self._rec_type_map[self._rec_type]["function"]
            attr = self._rec_type_map[self._rec_type]["rec_attr"]
            kwargs = self._rec_type_map[self._rec_type]["kwargs"]
            if kwargs:
                res = function(self._gpus, **kwargs)
            else:
                res = function(self._gpus)

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
            recording["max_values"] = [info.total for info in self._ctx.get_memory_info()]
        else:
            raise ValueError(f"Unsupported recording type: {self._rec_type}")

        return GPURecording(**recording)
