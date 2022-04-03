from dataclasses import dataclass
from math import floor

import numpy as np

_SEC = 1e9


class Recorder:
    def record(self, store):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def get_records(self):
        raise NotImplementedError


@dataclass
class GPUMetadata:
    device_idx: int
    device_name: str
    total_bytes: int


@dataclass
class GPURecording:
    """
    Stores recordings for a single GPU device.
    """
    metadata: GPUMetadata
    time_ns: np.ndarray
    data: np.ndarray

    @property
    def len(self):
        return len(self.time_ns)

    @property
    def duration(self):
        return (self.time_ns[-1] - self.time_ns[0]) / _SEC

    @property
    def sampling_rate(self):
        return floor(self.len / self.duration)
