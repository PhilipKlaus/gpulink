from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from pynvml import nvmlDeviceGetMemoryInfo

from .nvcontext import NVContext

_MB = 1e6
_SEC = 1e9


@dataclass
class GPUMemInfo:
    """
    Stores the memory usage for a single GPU device.
    """
    time_ns: int
    total_bytes: int
    used_bytes: int
    free_bytes: int


@dataclass
class GPUMemRecording:
    """
    Stores memory recordings for a single GPU device.
    """
    time_ns: np.ndarray
    total_bytes: np.ndarray
    used_bytes: np.ndarray


# ToDo: Store only used memory
# ToDo: Adapt dataclasses and str representation
class MemoryRecorder:
    """
    Manages GPU memory recording.
    """

    def __init__(self, ctx: NVContext):
        self._ctx = ctx
        self._memory_records = []
        self._time_ns = []
        self._total_bytes = []
        self._used_bytes = []
        self._free_bytes = []
        self.clear()

    def record(self) -> List[GPUMemInfo]:
        """
        Records and returns the actual GPU Memory usage.
        :return: A GPUMemRecord for each GPU in a List.
        """
        mem_info = self.memory_info
        for i, entry in enumerate(mem_info):
            self._memory_records[i]["time_ns"].append(entry.time_ns)
            self._memory_records[i]["total_bytes"].append(entry.total_bytes)
            self._memory_records[i]["used_bytes"].append(entry.used_bytes)
        return mem_info

    def clear(self) -> None:
        """
        Clear all previous records
        :return:
        """
        self._memory_records = [{"time_ns": [], "total_bytes": [], "used_bytes": []} for _ in range(self._ctx.gpus)]

    @property
    def memory_info(self) -> List[GPUMemInfo]:
        """
        Returns the actual GPU Memory usage without recording.
        :return: A GPUMemRecord for each GPU in a List.
        """
        mem_info = []
        for rec in self._ctx.execute_query(nvmlDeviceGetMemoryInfo):
            mem_info.append(GPUMemInfo(rec[0], rec[1].total, rec[1].used, rec[1].free))
        return mem_info

    @property
    def records(self) -> List[GPUMemRecording]:
        """
        Get all GPU memory recordings.
        :return: The memory recordings as a List of GPUMemRecording.
        """
        return [
            GPUMemRecording(np.array(r["time_ns"]), np.array(r["total_bytes"]), np.array(r["used_bytes"]))
            for r in self._memory_records]

    @property
    def num_records(self) -> int:
        """
        Get the number of stored records.
        :return: The number of stored records
        """
        return len(self.records[0].time_ns)

    def _generate_graph(self, show_total_mem=False) -> None:
        if self.num_records == 0:
            raise RuntimeError("No memory records available")

        max_mem = 0
        for i, rec_set in enumerate(self.records):
            max_mem = max(max_mem, np.max(rec_set.total_bytes))
            timestamps = (rec_set.time_ns - rec_set.time_ns[0]) / _SEC
            mem_usage = rec_set.used_bytes / _MB
            plt.plot(timestamps, mem_usage, label=f"GPU[{i}]")

        if show_total_mem:
            plt.ylim([0, max_mem / _MB])
        plt.legend(loc="upper left")
        plt.ylabel("Memory used [MB]")
        plt.xlabel("Time [s]")

    def save_graph(self, img_path: Path, show_total_mem=False) -> None:
        """
        Generates and saves a GPU memory usage graph.
        :param img_path: The path to the image file.
        :param show_total_mem: Show total memory on the y-axis.
        """
        self._generate_graph(show_total_mem)
        plt.savefig(img_path.as_posix())

    def draw_graph(self, show_total_mem=False) -> None:
        """
        Generates a GPU memory usage graph.
        :param show_total_mem: Scale y-axis to the total available GPU memory.
        """
        self._generate_graph(show_total_mem)
        plt.show()

    def __repr__(self) -> str:
        repr = f"MemoryRecorder: captured: {self.num_records} records\n"
        if self.num_records > 0:
            for i, data in enumerate(zip(self.records, self._ctx.gpu_names)):
                repr += f"GPU[{i}] ({data[1]}): min memory consumption: {np.min(data[0].used_bytes) / _MB}[MB]\n"
                repr += f"GPU[{i}] ({data[1]}): max memory consumption: {np.max(data[0].used_bytes) / _MB}[MB]\n"
        return repr
