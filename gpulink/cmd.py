import argparse
import logging
import signal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gpulink import MemoryPlotter
from .memory_recorder import MemoryRecorder
from .nvcontext import NVContext

_MB = 1e6

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
_LOGGER = logging.getLogger(__name__)


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
        _LOGGER.info(f"Start recording...")
        while _should_run:
            recorder.record()

    recordings = recorder.get_records()

    if args.output:
        graph = MemoryPlotter(recordings)
        graph.save(args.output, show_total_mem=True)

    _LOGGER.info(f"Measurement duration: {recordings[0].duration}[s]")
    _LOGGER.info(f"Average frame rate: {recordings[0].sampling_rate}[Hz]")
    for rec in recordings:
        idx = rec.device_idx
        name = rec.device_name
        _LOGGER.info(f"{name}[{idx}] -> memory used: "
                     f"min={np.min(rec.used_bytes) / _MB}[MB] / "
                     f"max={np.max(rec.used_bytes) / _MB}[MB]")


if __name__ == "__main__":
    main()
