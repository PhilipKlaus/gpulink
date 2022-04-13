from collections import namedtuple

import pytest

from gpulink import NVContext
from gpulink.types import GPUMemInfo, TemperatureThreshold, TemperatureSensorType, GPUQuerySingleResult

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
    mocker.patch("gpulink.nvcontext.time_ns", return_value=time)
    mocker.patch("gpulink.nvcontext.nvmlInit")
    mocker.patch("gpulink.nvcontext.nvmlShutdown")
    mocker.patch("gpulink.nvcontext.nvmlDeviceGetCount", return_value=2)
    mocker.patch("gpulink.nvcontext.nvmlDeviceGetHandleByIndex")
    mocker.patch("gpulink.nvcontext.nvmlDeviceGetName", return_value=b"GPU_TEST")
    mocker.patch("gpulink.nvcontext.nvmlDeviceGetMemoryInfo",
                 return_value=MemoryInfo(total=_GB, used=_GB // 2, free=_GB // 2))
    mocker.patch("gpulink.nvcontext.nvmlDeviceGetFanSpeed", return_value=_FAN_SPEED_PCT)
    mocker.patch("gpulink.nvcontext.nvmlDeviceGetFanSpeed_v2", return_value=_FAN_SPEED_PCT / 2)
    mocker.patch("gpulink.nvcontext.nvmlDeviceGetTemperature", return_value=_TMP)
    mocker.patch("gpulink.nvcontext.nvmlDeviceGetTemperatureThreshold", return_value=_TMP / 2)
    mocker.patch("gpulink.nvcontext.nvmlDeviceGetPowerUsage", return_value=_POWER_CONSUMPTION)


def test_nvcontext_properties_with_valid_ctx():
    with NVContext() as ctx:
        assert ctx.gpu_names == ["GPU_TEST", "GPU_TEST"]
        assert ctx.gpus == [0, 1]


def test_nvcontext_properties_with_invalid_ctx():
    ctx = NVContext()
    assert ctx.gpu_names == []
    assert ctx.gpus == []


def test_execute_raises_on_invalid_context():
    ctx = NVContext()
    with pytest.raises(RuntimeError, match="Cannot execute query in an invalid NVContext"):
        ctx.get_memory_info(None)


def test_get_memory_info():
    with NVContext() as ctx:
        assert ctx.get_memory_info(None) == [
            GPUMemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=0, timestamp=0, gpu_name="GPU_TEST"),
            GPUMemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_TEST")]
        assert ctx.get_memory_info(gpus=[1]) == [
            GPUMemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_TEST")
        ]


def test_get_fan_speed():
    with NVContext() as ctx:
        assert ctx.get_fan_speed(ctx.gpus, fan=0) == [
            GPUQuerySingleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_FAN_SPEED_PCT // 2),
            GPUQuerySingleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_FAN_SPEED_PCT // 2)
        ]


def test_get_temperature():
    with NVContext() as ctx:
        assert ctx.get_temperature(ctx.gpus, TemperatureSensorType.GPU) == [
            GPUQuerySingleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_TMP),
            GPUQuerySingleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP)
        ]


def test_get_temperature_threshold():
    with NVContext() as ctx:
        assert ctx.get_temperature_threshold(ctx.gpus, TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX) == [
            GPUQuerySingleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2),
            GPUQuerySingleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2)
        ]


def test_get_clock():
    with NVContext() as ctx:
        assert ctx.get_temperature_threshold(ctx.gpus, TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX) == [
            GPUQuerySingleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2),
            GPUQuerySingleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_TMP // 2)
        ]


def test_get_power_usage():
    with NVContext() as ctx:
        assert ctx.get_power_usage(ctx.gpus) == [
            GPUQuerySingleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_TEST", value=_POWER_CONSUMPTION),
            GPUQuerySingleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_TEST", value=_POWER_CONSUMPTION)
        ]
