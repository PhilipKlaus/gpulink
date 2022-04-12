from pathlib import Path

from matplotlib import pyplot as plt

from .consts import SEC, MB
from .types import GPURecording


class MemoryPlot:
    """
    Draws a GPU memory graph from a given GPUMemRecording.
    """

    def __init__(self, recording: GPURecording):
        self._recording = recording

    def _generate_graph(self, scale_y_axis) -> None:
        max_val = 0
        for gpu in self._recording.gpus:
            timestamps = self._recording.timestamps[gpu]
            data = self._recording.data[gpu]
            max_value = self._recording.max_values[gpu]
            gpu_name = self._recording.gpu_names[gpu]
            idx = self._recording.gpus[gpu]

            if timestamps.shape[0] == 0:
                raise RuntimeError("Recording data is empty")

            max_val = max(max_val, max_value)
            x_axis = (timestamps - timestamps[0]) / SEC
            y_axis = data / MB
            plt.plot(x_axis, y_axis, label=f"{gpu_name} [{idx}]")

        if not scale_y_axis:
            plt.ylim([0, max_val / MB])
        plt.legend(loc="upper left")
        plt.ylabel("Memory used [MB]")
        plt.xlabel("Time [s]")

    def save(self, img_path: Path, scale_y_axis=False) -> None:
        """
        Generates and saves a GPU memory graph.
        :param img_path: The path to the image file.
        :param scale_y_axis: Scale y-axis to the actual value range.
        """
        self._generate_graph(scale_y_axis)
        plt.savefig(img_path.as_posix())

    def plot(self, scale_y_axis=False) -> None:
        """
        Generates a GPU memory graph.
        :param scale_y_axis: Scale y-axis to the actual value range.
        """
        self._generate_graph(scale_y_axis)
        plt.show()
