import argparse
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pynvml import *

_MB = 1e6
_SEC = 1e9


@dataclass
class GPUMemRecord:
    """
    Stores the memory usage for a single GPU device.
    """
    timestamps: np.ndarray
    total: np.ndarray
    used: np.ndarray
    free: np.ndarray


class NVContext:
    """
    Context for wrapping nvmlInit() and nvmlShutdown().
    """

    def __enter__(self):
        nvmlInit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        nvmlShutdown()


class NVRecorder:
    """
    Manages GPU memory recording. Recordings must be performed in context of a NVContext.
    """

    def __init__(self, gpus: int):
        self._handles = []
        self.memory_records = [GPUMemRecord(np.array([]), np.array([]), np.array([]), np.array([])) for _ in
                               range(gpus)]
        for i in range(gpus):
            self._handles.append(nvmlDeviceGetHandleByIndex(i))

    def _execute_query(self, query, *args, **kwargs):
        res = []
        for handle in self._handles:
            timestamp = time.time_ns()
            res.append((timestamp, query(handle, *args, **kwargs)))
        return res

    def rec_memory_info(self) -> None:
        """
        Records the actual Memory usage in all GPU devices.
        :return: None
        """
        res = self._execute_query(nvmlDeviceGetMemoryInfo)
        for i, rec in enumerate(res):
            self.memory_records[i].timestamps = np.append(self.memory_records[i].timestamps, rec[0])
            self.memory_records[i].total = np.append(self.memory_records[i].total, rec[1].total)
            self.memory_records[i].used = np.append(self.memory_records[i].used, rec[1].used)
            self.memory_records[i].free = np.append(self.memory_records[i].free, rec[1].free)


def generate_memory_graph(img_path: Path, memory_records: List[GPUMemRecord]) -> None:
    """
    Generates and saves a plot of the GPU memory usage over time
    :param img_path: The path to the image file.
    :param memory_records: The memory_records to be plotted.
    :return: None
    """

    for i, rec_set in enumerate(memory_records):
        timestamps = (rec_set.timestamps - rec_set.timestamps[0]) / _SEC
        mem_usage = rec_set.used / _MB
        plt.plot(timestamps, mem_usage, label=f"GPU[{i}]")

    plt.legend(loc="upper left")
    plt.ylabel("Memory used [MB]")
    plt.xlabel("Time [s]")
    plt.savefig(img_path.as_posix())


def generate_cmd_output(memory_records: List[GPUMemRecord]) -> None:
    """
    Prints a summary of the measurements to the command line.
    :param memory_records: The memory_records to be summed up.
    :return: None
    """
    print(f"Measurement duration: {(memory_records[0].timestamps[-1] - memory_records[0].timestamps[0]) / _SEC} s]")
    for i, rec_set in enumerate(memory_records):
        print(f"GPU[0]: min memory consumption: {np.min(rec_set.used) / _MB}[MB]")
        print(f"GPU[0]: max memory consumption: {np.max(rec_set.used) / _MB}[MB]")


def _signal_handler(_, __):
    global _should_run
    _should_run = False


def _check_arguments(arguments):
    supported_file_types = plt.gcf().canvas.get_supported_filetypes()
    if args.output and arguments.output.suffix[1:] not in supported_file_types:
        raise ValueError(f"Output format '{arguments.output.suffix}' not supported")


_should_run = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-Link memory tracing")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Path to memory graph plot.")
    args = parser.parse_args()

    _check_arguments(args)

    signal.signal(signal.SIGINT, _signal_handler)
    with NVContext():
        deviceCount = nvmlDeviceGetCount()
        recorder = NVRecorder(deviceCount)
        while _should_run:
            recorder.rec_memory_info()

    recordings = recorder.memory_records
    generate_cmd_output(recordings)

    if args.output:
        generate_memory_graph(args.output, recordings)
