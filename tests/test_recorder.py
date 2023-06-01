import time

import numpy as np
import pytest

from gpulink import DeviceCtx, Recorder
from gpulink.devices.gpu import Gpu, GpuSet
from gpulink.recording.gpu_recording import RecType, Recording
from gpulink.recording.recorder import record
from gpulink.recording.timeseries import TimeSeries
from tests.device_mock import DeviceMock, TEST_GB


@pytest.fixture
def device_ctx():
    return DeviceCtx(device=DeviceMock)


def assert_timeseries_not_empty(recording: Recording):
    assert recording.timeseries[0].data.shape[0] > 0
    assert recording.timeseries[1].data.shape[0] > 0


def test_get_record(device_ctx):
    with device_ctx as ctx:
        rec = Recorder.create_memory_recorder(ctx, ctx.gpus.ids)
        assert rec._get_record() == ([0, 0], [TEST_GB // 2, TEST_GB // 2])


def test_fetch_and_return_data(device_ctx):
    with device_ctx as ctx:
        rec = Recorder.create_memory_recorder(ctx, ctx.gpus.ids)

        for i in range(3):
            rec._fetch_and_store()

        data = rec.get_recording()
        assert data.gpus == ctx.gpus
        assert data.rtype == RecType.REC_TYPE_MEMORY
        assert data.name == "GPULink Recording"

        assert data.timeseries == [
            TimeSeries(np.array([0, 0, 0]), np.array([TEST_GB // 2, TEST_GB // 2, TEST_GB // 2])),
            TimeSeries(np.array([0, 0, 0]), np.array([TEST_GB // 2, TEST_GB // 2, TEST_GB // 2]))
        ]


def test_record_using_start_stop(device_ctx):
    with device_ctx as ctx:
        rec = Recorder.create_memory_recorder(ctx, ctx.gpus.ids)

        # Ensure that the recording is empty at the beginning
        assert rec.get_recording().timeseries == [TimeSeries(np.array([]), np.array([])),
                                                  TimeSeries(np.array([]), np.array([]))]

        # Start recording -> wait for 1 second -> stop afterwards
        rec.start()
        time.sleep(1)
        rec.stop()

        assert_timeseries_not_empty(rec.get_recording())


def test_record_using_context_manager(device_ctx):
    with device_ctx as ctx:
        rec = Recorder.create_memory_recorder(ctx, ctx.gpus.ids)

        # Ensure that the recording is empty at the beginning
        recording = rec.get_recording()
        assert recording.timeseries == [TimeSeries(np.array([]), np.array([])), TimeSeries(np.array([]), np.array([]))]

        with rec:
            time.sleep(1)

        assert_timeseries_not_empty(rec.get_recording())


def test_record_using_record_decorator_default():
    @record(ctx_class=DeviceMock, rtype=RecType.REC_TYPE_CLOCK_GRAPHICS)
    def my_heavy_gpu_function(a: int, b: int):
        time.sleep(1)
        return a + b

    # Ensure that a recording is returned and that default values are set
    result = my_heavy_gpu_function(a=10, b=20)
    value = result.value
    assert value == 30
    recording = result.recording
    assert_timeseries_not_empty(recording)
    assert recording.rtype == RecType.REC_TYPE_CLOCK_GRAPHICS
    assert recording.name == "my_heavy_gpu_function"
    assert recording.gpus == GpuSet([Gpu(0, "GPU_0"), Gpu(1, "GPU_1")])
    assert recording.unit == "MHz"


def test_record_using_record_decorator_customized():
    name = "Decorator Test"

    @record(ctx_class=DeviceMock, rtype=RecType.REC_TYPE_CLOCK_GRAPHICS, name=name, gpus=[0])
    def my_heavy_gpu_function():
        return None

    # Ensure that a recording is returned and that customized parameters are set
    result = my_heavy_gpu_function()
    recording = result.recording
    assert recording.rtype == RecType.REC_TYPE_CLOCK_GRAPHICS
    assert recording.name == name
    assert recording.gpus == GpuSet([Gpu(0, "GPU_0")])
    assert recording.unit == "MHz"


unit_map = {
    RecType.REC_TYPE_CLOCK_GRAPHICS: "MHz",
    RecType.REC_TYPE_CLOCK_SM: "MHz",
    RecType.REC_TYPE_CLOCK_MEM: "MHz",
    RecType.REC_TYPE_CLOCK_VIDEO: "MHz",
    RecType.REC_TYPE_POWER_USAGE: "W",
    RecType.REC_TYPE_FAN_SPEED: "%",
    RecType.REC_TYPE_TEMPERATURE: "°C",
    RecType.REC_TYPE_MEMORY: "Byte"
}


def test_recording_with_all_rec_types(device_ctx):
    with device_ctx as ctx:
        for rec_type in RecType:
            recorder = Recorder.create_recorder(
                ctx=ctx,
                rtype=rec_type,
                name=rec_type.value,
            )

            with recorder:
                pass

            recording = recorder.get_recording()
            assert recording.gpus == GpuSet([Gpu(0, "GPU_0"), Gpu(1, "GPU_1")])
            assert recording.rtype == rec_type
            assert recording.name == rec_type.value
            assert recording.unit == unit_map[rec_type]
