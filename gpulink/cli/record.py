from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt

from gpulink import DeviceCtx, Recorder, Plot
from gpulink.cli.tools import busy_wait_for_interrupt
from gpulink.types import GPURecording


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


def record(args):
    """Record GPU properties"""

    _check_output_file_type(args.output)

    with DeviceCtx() as ctx:
        recorder = Recorder.create_memory_recorder(ctx, ctx.gpus)
        busy_wait_for_interrupt(recorder, "[RECORDING]")
        recording = recorder.get_recording()

    print(recording)

    _store_records(recording, args.output)
    _display_plot(recording, args.plot)
