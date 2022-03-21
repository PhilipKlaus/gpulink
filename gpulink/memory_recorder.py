from copy import deepcopy
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import List, Union

import numpy as np
from matplotlib import pyplot as plt
from pynvml import nvmlDeviceGetMemoryInfo

from .nvcontext import NVContext
from .recorder import Recorder

_MB = 1e6
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


@dataclass
class GPUMemRecording:
    """
    Stores memory recordings for a single GPU device.
    """
    device_idx: int
    device_name: str
    total_bytes: int
    time_ns: np.ndarray
    used_bytes: np.ndarray

    @property
    def len(self):
        return len(self.time_ns)

    @property
    def duration(self):
        return (self.time_ns[-1] - self.time_ns[0]) / _SEC

    @property
    def sampling_rate(self):
        return floor(self.len / self.duration)


class MemoryRecorder(Recorder):
    """
    Manages GPU memory recording.
    """

    def __init__(self, ctx: NVContext):
        self._ctx = ctx
        self._records: List[GPUMemRecording] = []
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
                self._records[idx].used_bytes.append(data.used)
        return mem_info

    def clear(self) -> None:
        """
        Clear all previous records
        """
        self._gpu_names = self._ctx.gpu_names
        records = self.record(store=False)
        self._records = []
        for rec in records:
            self._records.append(GPUMemRecording(rec.device_idx, rec.device_name, rec.total_bytes, [], []))

    def get_records(self) -> List[GPUMemRecording]:
        records = []
        for rec in self._records:
            new_rec = deepcopy(rec)
            new_rec.time_ns = np.array(new_rec.time_ns)
            new_rec.used_bytes = np.array(new_rec.used_bytes)
            records.append(new_rec)
        return records


class MemoryPlotter:
    """
    Draws a GPU memory graph from a given GPUMemRecording.
    """

    def __init__(self, recording: Union[GPUMemRecording, List[GPUMemRecording]]):
        self._recording = recording
        if not isinstance(self._recording, list):
            self._recording = [self._recording]

    def _generate_graph(self, show_total_mem=False) -> None:
        max_mem = 0
        for rec in self._recording:
            if rec.len == 0:
                raise RuntimeError("Memory recording is empty")

            max_mem = max(max_mem, rec.total_bytes)
            timestamps = (rec.time_ns - rec.time_ns[0]) / _SEC
            mem_usage = rec.used_bytes / _MB
            plt.plot(timestamps, mem_usage, label=f"GPU[{rec.device_idx}]")

        if show_total_mem:
            plt.ylim([0, max_mem / _MB])
        plt.legend(loc="upper left")
        plt.ylabel("Memory used [MB]")
        plt.xlabel("Time [s]")

    def save(self, img_path: Path, show_total_mem=False) -> None:
        """
        Generates and saves a GPU memory graph.
        :param img_path: The path to the image file.
        :param show_total_mem: Show total memory on the y-axis.
        """
        self._generate_graph(show_total_mem)
        plt.savefig(img_path.as_posix())

    def plot(self, show_total_mem=False) -> None:
        """
        Generates a GPU memory graph.
        :param show_total_mem: Scale y-axis to the total available GPU memory.
        """
        self._generate_graph(show_total_mem)
        plt.show()
