from pathlib import Path
from typing import Tuple, Union

from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure

from gpulink.consts import SEC
from gpulink.recording.gpu_recording import Recording


def _clean_matplotlib():
    plt.clf()
    plt.cla()
    plt.close()


DATA = Union[int, float]


class Plot:
    """
    Plots recorded GPU properties over time.
    """

    def __init__(self, recording: Recording):
        self._recording = recording

    def _describe_plot(self, ax):
        ax.set_title(self._recording.name)
        ax.legend(loc="upper left")
        ax.set_ylabel(f"{self._recording.rtype.value} [{self._recording.unit}]")
        ax.set_xlabel("Time [s]")

    def generate_graph(self) -> Tuple[Figure, Axis]:
        """
        Generates the plot.
        :return: A Tuple containing the generated Figure and Axis.
        """
        _clean_matplotlib()

        fig, ax = plt.subplots()

        for gpu, data in zip(self._recording.gpus, self._recording.timeseries):
            if data.timestamps.size == 0 or data.data.size == 0:
                raise ValueError("Timeseries data is empty")
            if data.timestamps.shape != data.data.shape:
                raise ValueError("Recorded timestamps and data must be of same shape")

            x_axis = (data.timestamps - data.timestamps[0]) / SEC
            y_axis = data.data

            ax.plot(x_axis, y_axis, label=f"{gpu.name} [{gpu.id}]")
            ax.autoscale()

        self._describe_plot(ax)
        return fig, ax

    def save(self, img_path: Path) -> None:
        """
        Generates and saves a Plot as an image.
        :param img_path: The path to the image file.
        """
        self.generate_graph()
        plt.savefig(img_path.as_posix())

    def plot(self) -> None:
        """
        Generates and display a Plot.
        """
        fig, _ = self.generate_graph()
        fig.canvas.manager.set_window_title("GPULink")
        plt.show()
