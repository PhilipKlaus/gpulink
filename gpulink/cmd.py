import argparse
import signal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gpulink import MemoryPlot
from .memory_recorder import MemoryRecorder
from .nvcontext import NVContext

_MB = 1e6


def _signal_handler(_, __):
    global _should_run
    _should_run = False


def _check_arguments(arguments):
    supported_file_types = plt.gcf().canvas.get_supported_filetypes()
    if arguments.output and arguments.output.suffix[1:] not in supported_file_types:
        raise ValueError(f"Output format '{arguments.output.suffix}' not supported")


_should_run = True


def main():
    parser = argparse.ArgumentParser(description="GPU-Link memory tracing")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Path to memory graph plot.")
    args = parser.parse_args()

    _check_arguments(args)

    signal.signal(signal.SIGINT, _signal_handler)
    with NVContext() as ctx:
        recorder = MemoryRecorder(ctx)
        print(f"Start recording...")
        while _should_run:
            recorder.record()

    recordings = recorder.get_records()

    if args.output:
        graph = MemoryPlot(recordings)
        graph.save(args.output, show_total_mem=True)

    print(f"Measurement duration: {recordings[0].duration}[s]")
    print(f"Average frame rate: {recordings[0].sampling_rate}[Hz]")
    for rec in recordings:
        idx = rec.metadata.device_idx
        name = rec.metadata.device_name
        print(f"{name}[{idx}] -> memory used: "
              f"min={np.min(rec.data) / _MB}[MB] / "
              f"max={np.max(rec.data) / _MB}[MB]")


if __name__ == "__main__":
    main()
