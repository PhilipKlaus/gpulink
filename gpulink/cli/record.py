from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt

from gpulink import Plot
from gpulink import NVContext, Recorder
from gpulink.cli.tools import busy_wait_for_interrupt
from gpulink.types import GPURecording, RecType


def _check_output_file_type(output_path: Path):
    supported_file_types = plt.gcf().canvas.get_supported_filetypes()
    if output_path and output_path.suffix[1:] not in supported_file_types:
        raise ValueError(f"Output format '{output_path.suffix}' not supported")


def _store_records(recording: GPURecording, output_path: Optional[Path]):
    if output_path:
        graph = Plot(recording)
        graph.save(output_path, scale_y_axis=True)


def record(args):
    """Record GPU properties"""

    _check_output_file_type(args.output)

    with NVContext() as ctx:
        recorder = Recorder.create_memory_recorder(ctx, ctx.gpus)
        busy_wait_for_interrupt(recorder, "[RECORDING]")
        recording = recorder.get_recording()

    _store_records(recording, args.output)
    print(recording)
