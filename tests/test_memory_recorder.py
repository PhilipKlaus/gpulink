import time
from dataclasses import dataclass

from pynvml import nvmlDeviceGetMemoryInfo

from gpulink import MemoryRecorder, GPUMemInfo
from gpulink.nvcontext import Result

_SEC = 1e9
GB = 1000000000
TS = time.time_ns()


@dataclass
class MemoryInfoMock:
    total: int
    used: int
    free: int


class CtxMock:
    gpu_names = ["GPU1", "GPU2"]

    def execute_query(self, query):
        if query == nvmlDeviceGetMemoryInfo:
            return [
                Result(timestamp=TS, data=MemoryInfoMock(GB, GB // 2, GB // 2)),
                Result(timestamp=TS, data=MemoryInfoMock(GB, GB // 2, GB // 2)),
            ]


def test_record_returns_gpu_memory_info():
    recorder = MemoryRecorder(CtxMock())
    record = recorder.record(store=False)
    assert len(record) == 2
    assert record[0] == GPUMemInfo(device_idx=0, device_name="GPU1", time_ns=TS, total_bytes=GB, used_bytes=GB // 2,
                                   free_bytes=GB // 2)
    assert record[1] == GPUMemInfo(device_idx=1, device_name="GPU2", time_ns=TS, total_bytes=GB, used_bytes=GB // 2,
                                   free_bytes=GB // 2)


def test_record_stores():
    recorder = MemoryRecorder(CtxMock())

    # Check records before storing
    records = recorder.get_records()
    assert len(records) == 2
    assert records[0].time_ns.shape == (0,)
    assert records[0].data.shape == (0,)

    # Record
    for i in range(5):
        recorder.record()

    # Check if records are present
    records = recorder.get_records()
    assert records[0].time_ns.shape == (5,)
    assert records[0].data.shape == (5,)


def test_clear():
    recorder = MemoryRecorder(CtxMock())

    # Record
    for i in range(5):
        recorder.record()

    recorder.clear()

    # Check that records are cleared
    records = recorder.get_records()
    assert records[0].time_ns.shape == (0,)
    assert records[0].data.shape == (0,)
