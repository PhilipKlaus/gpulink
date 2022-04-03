import time

import numpy as np

from gpulink.recorder import GPURecording, GPUMetadata

_SEC = 1e9
_GB = 1000000000


def test_gpu_mem_recording():
    ts = time.time()
    time_ns = []
    data = []

    for i in range(10):
        time_ns.append(ts)
        data.append(1)
        ts = ts + _SEC // 2

    recording = GPURecording(GPUMetadata(device_idx=0, device_name="GPU0", total_bytes=_GB), np.array(time_ns),
                             np.array(data))
    assert recording.len == 10
    assert recording.duration == 4.5
    assert recording.sampling_rate == 2
