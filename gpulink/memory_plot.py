from pathlib import Path
from typing import Union, List

from matplotlib import pyplot as plt

from .recorder import GPURecording

_MB = 1e6
_SEC = 1e9


class MemoryPlot:
    """
    Draws a GPU memory graph from a given GPUMemRecording.
    """

    def __init__(self, recording: Union[GPURecording, List[GPURecording]]):
        self._recording = recording
        if not isinstance(self._recording, list):
            self._recording = [self._recording]

    def _generate_graph(self, show_total_mem=False) -> None:
        max_mem = 0
        for rec in self._recording:
            if rec.len == 0:
                raise RuntimeError("Memory recording is empty")

            max_mem = max(max_mem, rec.metadata.total_bytes)
            timestamps = (rec.time_ns - rec.time_ns[0]) / _SEC
            mem_usage = rec.data / _MB
            plt.plot(timestamps, mem_usage, label=f"GPU[{rec.metadata.device_idx}]")

        if show_total_mem:
            plt.ylim([0, max_mem / _MB])
        plt.legend(loc="upper left")
        plt.ylabel("Memory used [MB]")
        plt.xlabel("Time [s]")

    def save(self, img_path: Path, show_total_mem=True) -> None:
        """
        Generates and saves a GPU memory graph.
        :param img_path: The path to the image file.
        :param show_total_mem: Show total available memory on the y-axis.
        """
        self._generate_graph(show_total_mem)
        plt.savefig(img_path.as_posix())

    def plot(self, show_total_mem=True) -> None:
        """
        Generates a GPU memory graph.
        :param show_total_mem: Scale y-axis to the total available GPU memory.
        """
        self._generate_graph(show_total_mem)
        plt.show()
