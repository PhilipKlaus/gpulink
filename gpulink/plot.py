from pathlib import Path
from typing import Tuple

from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure

from .consts import SEC
from .types import GPURecording


def _clean_matplotlib():
    plt.clf()
    plt.cla()
    plt.close()


class Plot:
    """
    Plots recorded GPU properties over time.
    """

    def __init__(self, recording: GPURecording):
        self._recording = recording

    def _describe_plot(self, ax):
        ax.set_title(f"GPULink Recording: {self._recording.type.value[0]}")
        ax.legend(loc="upper left")
        ax.set_ylabel(f"{self._recording.type.value[0]} [{self._recording.type.value[1].name}]")
        ax.set_xlabel("Time [s]")

    def generate_graph(self, scale_y_axis) -> Tuple[Figure, Axis]:
        """
        Generates the plot.
        :param scale_y_axis: Scale y-axis to the actual value range. E.g. in case of plotting memory consumption this
        means that the y-axis is scaled to the actual consumed memory and not to the maximum available memory.
        :return: A Tuple containing the generated Figure and Axis.
        """
        _clean_matplotlib()
        unit_divider = self._recording.type.value[1].value
        max_val = 0

        fig, ax = plt.subplots()

        for gpu in self._recording.gpus:
            timestamps = self._recording.timestamps[gpu]
            data = self._recording.data[gpu]
            max_value = self._recording.plot_info.max_values[gpu]
            gpu_name = self._recording.gpu_names[gpu]
            idx = self._recording.gpus[gpu]

            if timestamps.shape[0] == 0:
                raise RuntimeError("Recording data is empty")

            max_val = max(max_val, max_value)
            x_axis = (timestamps - timestamps[0]) / SEC
            y_axis = data / unit_divider

            ax.plot(x_axis, y_axis, label=f"{gpu_name} [{idx}]")

        if not scale_y_axis:
            plt.ylim([0, max_val / unit_divider])

        self._describe_plot(ax)
        return fig, ax

    def save(self, img_path: Path, scale_y_axis=False) -> None:
        """
        Generates and saves a Plot as an image.
        :param img_path: The path to the image file.
        :param scale_y_axis: Scale y-axis to the actual value range. E.g. in case of plotting memory consumption this
        means that the y-axis is scaled to the actual consumed memory and not to the maximum available memory.
        """
        self.generate_graph(scale_y_axis)
        plt.savefig(img_path.as_posix())

    def plot(self, scale_y_axis=False) -> None:
        """
        Generates and display a Plot.
        :param scale_y_axis: Scale y-axis to the actual value range. E.g. in case of plotting memory consumption this
        means that the y-axis is scaled to the actual consumed memory and not to the maximum available memory.
        """
        self.generate_graph(scale_y_axis)
        plt.show()
