from copy import deepcopy
from dataclasses import dataclass
from typing import List

import numpy as np
from pynvml import nvmlDeviceGetMemoryInfo

from .nvcontext import NVContext
from .recorder import Recorder, GPURecording, GPUMetadata

_SEC = 1e9


@dataclass
class GPUMemInfo:
    """
    Stores memory information for a single GPU device.
    """
    device_idx: int
    device_name: str
    time_ns: int
    total_bytes: int
    used_bytes: int
    free_bytes: int


class MemoryRecorder(Recorder):
    """
    Manages GPU memory recording.
    """

    def __init__(self, ctx: NVContext):
        self._ctx = ctx
        self._records: List[GPURecording] = []
        self._gpu_names: List[str] = []
        self.clear()

    def record(self, store=True) -> List[GPUMemInfo]:
        """
        Records and returns the actual GPU Memory usage.
        :param store: Specify if the record shall be stored (default=True)
        :return: A GPUMemRecord for each GPU in a List.
        """
        mem_info = []
        for idx, res in enumerate(self._ctx.execute_query(nvmlDeviceGetMemoryInfo)):
            timestamp = res.timestamp
            data = res.data
            mem_info.append(GPUMemInfo(idx, self._gpu_names[idx], timestamp, data.total, data.used, data.free))
            if store:
                self._records[idx].time_ns.append(timestamp)
                self._records[idx].data.append(data.used)
        return mem_info

    def clear(self) -> None:
        """
        Clear all previous records
        """
        self._gpu_names = self._ctx.gpu_names
        records = self.record(store=False)
        self._records = []
        for rec in records:
            self._records.append(GPURecording(GPUMetadata(rec.device_idx, rec.device_name, rec.total_bytes), [], []))

    def get_records(self) -> List[GPURecording]:
        """
        Get a copy of all stored records in a list of GPUMemRecording.
        :return: The stored records.
        """
        records = []
        for rec in self._records:
            new_rec = deepcopy(rec)
            new_rec.time_ns = np.array(new_rec.time_ns)
            new_rec.data = np.array(new_rec.data)
            records.append(new_rec)
        return records
