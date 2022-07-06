from collections import namedtuple

import pytest

from gpulink import DeviceCtx
from gpulink.types import MemInfo, TemperatureThreshold, TemperatureSensorType, SimpleResult, Gpu, GpuSet

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
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetFanSpeed_v2", return_value=_FAN_SPEED_PCT / 2)
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetTemperature", return_value=_TMP)
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetTemperatureThreshold", return_value=_TMP / 2)
    mocker.patch("gpulink.devices.nvml_device.nvmlDeviceGetPowerUsage", return_value=_POWER_CONSUMPTION)


def test_nvcontext_properties_with_valid_ctx():
    with DeviceCtx() as ctx:
        assert ctx.gpus == GpuSet([Gpu(0, "GPU_TEST"), Gpu(1, "GPU_TEST")])


def test_execute_raises_on_invalid_context():
    ctx = DeviceCtx()
    with pytest.raises(RuntimeError, match="Cannot execute query in an invalid NVContext"):
        ctx.get_memory_info(None)


def test_get_memory_info():
    with DeviceCtx() as ctx:
        assert ctx.get_memory_info(None) == [
            MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=0, timestamp=0, gpu_name="GPU_TEST"),
            MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_TEST")]
        assert ctx.get_memory_info(gpus=[1]) == [
            MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_TEST")
        ]


def test_get_fan_speed():
    with DeviceCtx() as ctx:
        assert ctx.get_fan_speed(ctx.gpus.ids, fan=0) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_FAN_SPEED_PCT // 2),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_FAN_SPEED_PCT // 2)
        ]


def test_get_temperature():
    with DeviceCtx() as ctx:
        assert ctx.get_temperature(ctx.gpus.ids, TemperatureSensorType.GPU) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_TMP),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP)
        ]


def test_get_temperature_threshold():
    with DeviceCtx() as ctx:
        assert ctx.get_temperature_threshold(ctx.gpus.ids, TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2)
        ]


def test_get_clock():
    with DeviceCtx() as ctx:
        assert ctx.get_temperature_threshold(ctx.gpus.ids, TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2)
        ]


def test_get_power_usage():
    with DeviceCtx() as ctx:
        assert ctx.get_power_usage(ctx.gpus.ids) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_POWER_CONSUMPTION),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_POWER_CONSUMPTION)
        ]
