"""
Tests for the LocalNvmlGpu using patched nvml functions
"""

from collections import namedtuple

import pytest

from gpulink import DeviceCtx
from gpulink.gpu_types import MemInfo, TemperatureThreshold, TemperatureSensorType, SimpleResult, Gpu, GpuSet, ClockType, \
    ClockId

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
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetName", return_value=b"GPU_TEST")
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
    with DeviceCtx() as ctx:
        assert ctx.gpus == GpuSet([Gpu(0, "GPU_TEST"), Gpu(1, "GPU_TEST")])


def test_get_memory_info():
    with DeviceCtx() as ctx:
        assert ctx.get_memory_info() == [
            MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=0, timestamp=0, gpu_name="GPU_TEST"),
            MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_TEST")]
        assert ctx.get_memory_info(gpus=[1]) == [
            MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_TEST")
        ]


def test_get_fan_speed():
    with DeviceCtx() as ctx:
        assert ctx.get_fan_speed(fan=0) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_FAN_SPEED_PCT // 2),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_FAN_SPEED_PCT // 2)
        ]
        assert ctx.get_fan_speed(gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_FAN_SPEED_PCT)
        ]


def test_get_temperature():
    with DeviceCtx() as ctx:
        assert ctx.get_temperature(TemperatureSensorType.GPU) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_TMP),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP)
        ]
        assert ctx.get_temperature(TemperatureSensorType.GPU, gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP)
        ]


def test_get_temperature_threshold():
    with DeviceCtx() as ctx:
        assert ctx.get_temperature_threshold(TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2)
        ]
        assert ctx.get_temperature_threshold(TemperatureThreshold.TEMPERATURE_THRESHOLD_COUNT, gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2),
        ]


def test_get_clock():
    with DeviceCtx() as ctx:
        assert ctx.get_clock(ClockType.CLOCK_SM) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_CLOCK),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_CLOCK)
        ]
        assert ctx.get_clock(ClockType.CLOCK_SM, clock_id=ClockId.CLOCK_ID_CURRENT, gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_CLOCK // 2)
        ]


def test_get_power_usage():
    with DeviceCtx() as ctx:
        assert ctx.get_power_usage(ctx.gpus.ids) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_POWER_CONSUMPTION),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_POWER_CONSUMPTION)
        ]
