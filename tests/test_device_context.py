"""
Tests for the DeviceCtx using a mocked device.
"""

import pytest

import gpulink as gpu
from gpulink.devices.device_mock import TEST_GB, TEST_FAN_SPEED_PCT, TEST_TEMP, TEST_CLOCK, TEST_POWER_CONSUMPTION


@pytest.fixture
def device_ctx():
    return gpu.DeviceCtx(device=gpu.DeviceMock)


def test_valid_ctx(device_ctx):
    assert not device_ctx.valid_ctx
    with device_ctx as ctx:
        assert ctx.valid_ctx


def test_raises_on_invalid_context(device_ctx):
    with pytest.raises(RuntimeError, match="Cannot execute query in an invalid NVContext"):
        device_ctx.get_memory_info()


def test_gpus(device_ctx):
    with device_ctx as ctx:
        assert ctx.gpus == gpu.GpuSet([gpu.Gpu(0, "GPU_0"), gpu.Gpu(1, "GPU_1")])


def test_get_memory_info(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_memory_info() == [
            gpu.MemInfo(total=TEST_GB, used=TEST_GB // 2, free=TEST_GB // 2, gpu_idx=0, timestamp=0,
                        gpu_name="GPU_0"),
            gpu.MemInfo(total=TEST_GB, used=TEST_GB // 4, free=TEST_GB // 4, gpu_idx=1, timestamp=0,
                        gpu_name="GPU_1")]
        assert ctx.get_memory_info(gpus=[1]) == [
            gpu.MemInfo(total=TEST_GB, used=TEST_GB // 4, free=TEST_GB // 4, gpu_idx=1, timestamp=1,
                        gpu_name="GPU_1")
        ]


def test_get_fan_speed(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_fan_speed(fan=0) == [
            gpu.SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_FAN_SPEED_PCT // 2),
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_FAN_SPEED_PCT // 2)
        ]
        assert ctx.get_fan_speed(fan=1, gpus=[1]) == [
            gpu.SimpleResult(gpu_idx=1, timestamp=1, gpu_name="GPU_1", value=TEST_FAN_SPEED_PCT // 4)
        ]


def test_get_temperature(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_temperature(gpu.TemperatureSensorType.GPU) == [
            gpu.SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_TEMP),
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_TEMP)
        ]
        assert ctx.get_temperature(gpu.TemperatureSensorType.GPU, gpus=[1]) == [
            gpu.SimpleResult(gpu_idx=1, timestamp=1, gpu_name="GPU_1", value=TEST_TEMP)
        ]


def test_get_temperature_threshold(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_temperature_threshold(gpu.TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX) == [
            gpu.SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_TEMP // 2),
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_TEMP // 2)
        ]
        assert ctx.get_temperature_threshold(gpu.TemperatureThreshold.TEMPERATURE_THRESHOLD_COUNT, gpus=[1]) == [
            gpu.SimpleResult(gpu_idx=1, timestamp=1, gpu_name="GPU_1", value=TEST_TEMP // 4),
        ]


def test_get_clock(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_clock(gpu.ClockType.CLOCK_SM) == [
            gpu.SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_CLOCK),
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_CLOCK)
        ]
        assert ctx.get_clock(gpu.ClockType.CLOCK_SM, clock_id=gpu.ClockId.CLOCK_ID_CURRENT, gpus=[1]) == [
            gpu.SimpleResult(gpu_idx=1, timestamp=1, gpu_name="GPU_1", value=TEST_CLOCK // 2)
        ]


def test_get_power_usage(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_power_usage(ctx.gpus.ids) == [
            gpu.SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_POWER_CONSUMPTION),
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_POWER_CONSUMPTION)
        ]
        assert ctx.get_power_usage(gpus=[1]) == [
            gpu.SimpleResult(gpu_idx=1, timestamp=1, gpu_name="GPU_1", value=TEST_POWER_CONSUMPTION)
        ]
