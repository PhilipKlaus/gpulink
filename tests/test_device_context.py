"""
Tests for the DeviceCtx using a mocked device.
"""

import pytest

from gpulink import ClockType, ClockId, DeviceCtx
from gpulink.gpu_types import MemInfo, TemperatureThreshold, TemperatureSensorType, SimpleResult, Gpu, GpuSet
from tests.misc import DeviceMock, TEST_CLOCK, TEST_POWER_CONSUMPTION, TEST_GB, TEST_FAN_SPEED_PCT, TEST_TEMP


@pytest.fixture
def device_ctx():
    return DeviceCtx(device=DeviceMock)


def test_valid_ctx(device_ctx):
    assert not device_ctx.valid_ctx
    with device_ctx as ctx:
        assert ctx.valid_ctx


def test_raises_on_invalid_context(device_ctx):
    with pytest.raises(RuntimeError, match="Cannot execute query in an invalid NVContext"):
        device_ctx.get_memory_info()


def test_gpus(device_ctx):
    with device_ctx as ctx:
        assert ctx.gpus == GpuSet([Gpu(0, "GPU_0"), Gpu(1, "GPU_1")])


def test_get_memory_info(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_memory_info() == [
            MemInfo(total=TEST_GB, used=TEST_GB // 2, free=TEST_GB // 2, gpu_idx=0, timestamp=0, gpu_name="GPU_0"),
            MemInfo(total=TEST_GB, used=TEST_GB // 2, free=TEST_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_1")]
        assert ctx.get_memory_info(gpus=[1]) == [
            MemInfo(total=TEST_GB, used=TEST_GB // 2, free=TEST_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_1")
        ]


def test_get_fan_speed(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_fan_speed(fan=0) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_FAN_SPEED_PCT // 2),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_FAN_SPEED_PCT // 2)
        ]
        assert ctx.get_fan_speed(fan=1, gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_FAN_SPEED_PCT // 4)
        ]


def test_get_temperature(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_temperature(TemperatureSensorType.GPU) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_TEMP),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_TEMP)
        ]
        assert ctx.get_temperature(TemperatureSensorType.GPU, gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_TEMP)
        ]


def test_get_temperature_threshold(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_temperature_threshold(TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_TEMP // 2),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_TEMP // 2)
        ]
        assert ctx.get_temperature_threshold(TemperatureThreshold.TEMPERATURE_THRESHOLD_COUNT, gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_TEMP // 4),
        ]


def test_get_clock(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_clock(ClockType.CLOCK_SM) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_CLOCK),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_CLOCK)
        ]
        assert ctx.get_clock(ClockType.CLOCK_SM, clock_id=ClockId.CLOCK_ID_CURRENT, gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_CLOCK // 2)
        ]


def test_get_power_usage(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_power_usage(ctx.gpus.ids) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_POWER_CONSUMPTION),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_POWER_CONSUMPTION)
        ]
        assert ctx.get_power_usage(gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_POWER_CONSUMPTION)
        ]
