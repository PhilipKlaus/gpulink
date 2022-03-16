import argparse
import signal
from math import floor
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from .memory_recorder import MemoryRecorder, GPUMemRecording
from .nvcontext import NVContext


def _signal_handler(_, __):
    global _should_run
    _should_run = False


def _check_arguments(arguments):
    supported_file_types = plt.gcf().canvas.get_supported_filetypes()
    if arguments.output and arguments.output.suffix[1:] not in supported_file_types:
        raise ValueError(f"Output format '{arguments.output.suffix}' not supported")


def _get_measurement_duration(mem_records: List[GPUMemRecording]):
    return (mem_records[0].time_ns[-1] - mem_records[0].time_ns[0]) / 1e9


_should_run = True


def main():
    parser = argparse.ArgumentParser(description="GPU-Link memory tracing")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Path to memory graph plot.")
    args = parser.parse_args()

    _check_arguments(args)

    signal.signal(signal.SIGINT, _signal_handler)
    with NVContext() as ctx:
        recorder = MemoryRecorder(ctx)
        while _should_run:
            recorder.record()

    if args.output:
        recorder.save_graph(args.output)

    duration = _get_measurement_duration(recorder.records)
    print(f"Measurement duration: {duration}[s]")
    print(f"Average frame rate: {floor(recorder.num_records / duration)}[Hz]")
    print(recorder)


if __name__ == "__main__":
    main()
