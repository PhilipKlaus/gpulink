from pathlib import Path
from typing import Optional, Callable, List, Union

from matplotlib import pyplot as plt

from gpulink import DeviceCtx, Plot, Recorder
from gpulink.cli.tools import start_in_background
from gpulink.factory import factory, make
from gpulink.stoppable_thread import StoppableThread
from gpulink.types import GPURecording, QueryResult


def _check_output_file_type(output_path: Path):
    supported_file_types = plt.gcf().canvas.get_supported_filetypes()
    # Necessary to ensure that implicitly created figure is deleted
    plt.clf()
    plt.cla()
    plt.close()
    if output_path and output_path.suffix[1:] not in supported_file_types:
        raise ValueError(f"Output format '{output_path.suffix}' not supported")


def _store_records(recording: GPURecording, output_path: Optional[Path]):
    if output_path:
        graph = Plot(recording)
        graph.save(output_path, scale_y_axis=True)


def _display_plot(recording: GPURecording, plot: bool):
    if plot:
        p = Plot(recording)
        p.plot(scale_y_axis=True)


"""
@factory
class Recorder(StoppableThread):

    def __init__(self, cmd: Callable[[], List[QueryResult]], filter: Callable[[QueryResult], Union[int, float, str]]):
        super().__init__()
        self._cmd = cmd
        self._filter = filter
        self._data = []

    @classmethod
    @make
    def create_memory_recorder(cls, key, ctx: DeviceCtx, gpus: List[int]):
        return Recorder(
            key,
            cmd=lambda: ctx.get_memory_info(gpus),
            filter=lambda res: res.used
        )

    def fetch_data(self):
        data = []
        for result in self._cmd():
            data.append(self._filter(result))
        return data

    def run(self):
        while not self.should_stop:
            data.append(self.fetch_data())

    def get_recording(self):
        return None
"""


def record(args):
    """Record GPU properties"""

    _check_output_file_type(args.output)

    with DeviceCtx() as ctx:
        recorder = Recorder.create_memory_recorder(ctx, ctx.gpus.ids)
        start_in_background(recorder, "[RECORDING]")
        recording = recorder.get_recording()

    print("")
    print(recording)

    # _store_records(recording, args.output)
    # _display_plot(recording, args.plot)
