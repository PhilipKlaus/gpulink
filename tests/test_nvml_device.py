"""
Tests for the LocalNvmlGpu using patched nvml functions
"""

from collections import namedtuple

import pytest

import gpulink as gpu

_GB = int(1e9)

MemoryInfo = namedtuple('MemoryInfo', 'total used free')
FanSpeed = namedtuple('FanSpeed', 'speed')
Temperature = namedtuple('Temperature', 'temperature')

time = 0

_FAN_SPEED_PCT = 100
_TMP = 30
_CLOCK = 100
_POWER_CONSUMPTION = 30


@pytest.fixture(autouse=True)
def patch_nvcontext(mocker):
    time = 0
    mocker.patch("gpulink.devices.nvml_device.time_ns", return_value=time)
    mocker.patch("gpulink.devices.nvml_device.nvmlInit")
    mocker.patch("gpulink.devices.nvml_device.nvmlShutdown")
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetCount", return_value=2)
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetHandleByIndex")
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetName", return_value="GPU_TEST")
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetMemoryInfo",
                 return_value=MemoryInfo(total=_GB, used=_GB // 2, free=_GB // 2))
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetFanSpeed", return_value=_FAN_SPEED_PCT)
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetFanSpeed_v2", return_value=_FAN_SPEED_PCT // 2)
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetTemperature", return_value=_TMP)
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetTemperatureThreshold", return_value=_TMP // 2)
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetClock", return_value=_CLOCK // 2)
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetClockInfo", return_value=_CLOCK)
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetPowerUsage", return_value=_POWER_CONSUMPTION)


def test_gpus():
    with gpu.DeviceCtx() as ctx:
        assert ctx.gpus == gpu.GpuSet([gpu.Gpu(0, "GPU_TEST"), gpu.Gpu(1, "GPU_TEST")])


def test_get_memory_info():
    with gpu.DeviceCtx() as ctx:
        assert ctx.get_memory_info() == [
            gpu.MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=0, timestamp=0, gpu_name="GPU_TEST"),
            gpu.MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_TEST")]
        assert ctx.get_memory_info(gpus=[1]) == [
            gpu.MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_TEST")
        ]


def test_get_fan_speed():
    with gpu.DeviceCtx() as ctx:
        assert ctx.get_fan_speed(fan=0) == [
            gpu.SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_FAN_SPEED_PCT // 2),
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_FAN_SPEED_PCT // 2)
        ]
        assert ctx.get_fan_speed(gpus=[1]) == [
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_FAN_SPEED_PCT)
        ]


def test_get_temperature():
    with gpu.DeviceCtx() as ctx:
        assert ctx.get_temperature(gpu.TemperatureSensorType.GPU) == [
            gpu.SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_TMP),
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP)
        ]
        assert ctx.get_temperature(gpu.TemperatureSensorType.GPU, gpus=[1]) == [
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP)
        ]


def test_get_temperature_threshold():
    with gpu.DeviceCtx() as ctx:
        assert ctx.get_temperature_threshold(gpu.TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX) == [
            gpu.SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2),
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2)
        ]
        assert ctx.get_temperature_threshold(gpu.TemperatureThreshold.TEMPERATURE_THRESHOLD_COUNT, gpus=[1]) == [
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2),
        ]


def test_get_clock():
    with gpu.DeviceCtx() as ctx:
        assert ctx.get_clock(gpu.ClockType.CLOCK_SM) == [
            gpu.SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_CLOCK),
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_CLOCK)
        ]
        assert ctx.get_clock(gpu.ClockType.CLOCK_SM, clock_id=gpu.ClockId.CLOCK_ID_CURRENT, gpus=[1]) == [
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_CLOCK // 2)
        ]


def test_get_power_usage():
    with gpu.DeviceCtx() as ctx:
        assert ctx.get_power_usage(ctx.gpus.ids) == [
            gpu.SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_POWER_CONSUMPTION),
            gpu.SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_POWER_CONSUMPTION)
        ]
