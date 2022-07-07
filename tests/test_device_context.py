"""
Tests for the DeviceCtx using a mocked device.
"""
from typing import Optional, List

import pytest

from gpulink import DeviceCtx, ClockType, ClockId
from gpulink.devices.base_device import BaseDevice
from gpulink.types import MemInfo, TemperatureThreshold, TemperatureSensorType, SimpleResult, Gpu, GpuSet

_GB = int(1e9)
_FAN_SPEED_PCT = 100
_TMP = 30
_CLOCK = 100
_POWER_CONSUMPTION = 30


class DeviceMock(BaseDevice):

    def setup(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_gpus(self) -> GpuSet:
        return GpuSet([Gpu(0, "GPU_0"), Gpu(1, "GPU_1")])

    def get_memory_info(self, gpus: Optional[List[int]] = None) -> List[MemInfo]:
        infos = [MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=0, timestamp=0, gpu_name="GPU_0"),
                 MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_1")]
        return infos if not gpus else [infos[i] for i in gpus]

    def get_fan_speed(self, fan=None, gpus: Optional[List[int]] = None) -> List[SimpleResult]:
        if not fan or fan == 0:
            speed = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_FAN_SPEED_PCT // 2),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_FAN_SPEED_PCT // 2)
            ]
        else:
            speed = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_FAN_SPEED_PCT // 4),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_FAN_SPEED_PCT // 4)
            ]
        return speed if not gpus else [speed[i] for i in gpus]

    def get_temperature(self, sensor_type: TemperatureSensorType, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        temps = [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_TMP),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_TMP)
        ]
        return temps if not gpus else [temps[i] for i in gpus]

    def get_temperature_threshold(self, threshold: TemperatureThreshold, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        if threshold == TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX:
            temps = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_TMP // 2),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_TMP // 2)
            ]
        else:
            temps = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_TMP // 4),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_TMP // 4)
            ]
        return temps if not gpus else [temps[i] for i in gpus]

    def get_clock(self, clock_type: ClockType, clock_id: ClockId = None, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:

        if not clock_id or clock_id == ClockId.CLOCK_ID_APP_CLOCK_DEFAULT:
            clock = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_CLOCK),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_CLOCK)
            ]
        else:
            clock = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_CLOCK // 2),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_CLOCK // 2)
            ]
        return clock if not gpus else [clock[i] for i in gpus]

    def get_power_usage(self, gpus: Optional[List[int]]) -> List[SimpleResult]:
        power = [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_POWER_CONSUMPTION),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_POWER_CONSUMPTION)
        ]
        return power if not gpus else [power[i] for i in gpus]


@pytest.fixture
def device_ctx():
    return DeviceCtx(device_type=DeviceMock)


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
            MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=0, timestamp=0, gpu_name="GPU_0"),
            MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_1")]
        assert ctx.get_memory_info(gpus=[1]) == [
            MemInfo(total=_GB, used=_GB // 2, free=_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_1")
        ]


def test_get_fan_speed(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_fan_speed(fan=0) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_FAN_SPEED_PCT // 2),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_FAN_SPEED_PCT // 2)
        ]
        assert ctx.get_fan_speed(fan=1, gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_FAN_SPEED_PCT // 4)
        ]


def test_get_temperature(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_temperature(TemperatureSensorType.GPU) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_TMP),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_TMP)
        ]
        assert ctx.get_temperature(TemperatureSensorType.GPU, gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_TMP)
        ]


def test_get_temperature_threshold(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_temperature_threshold(TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_TMP // 2),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_TMP // 2)
        ]
        assert ctx.get_temperature_threshold(TemperatureThreshold.TEMPERATURE_THRESHOLD_COUNT, gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_TMP // 4),
        ]


def test_get_clock(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_clock(ClockType.CLOCK_SM) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_CLOCK),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_CLOCK)
        ]
        assert ctx.get_clock(ClockType.CLOCK_SM, clock_id=ClockId.CLOCK_ID_CURRENT, gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_CLOCK // 2)
        ]


def test_get_power_usage(device_ctx):
    with device_ctx as ctx:
        assert ctx.get_power_usage(ctx.gpus.ids) == [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=_POWER_CONSUMPTION),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_POWER_CONSUMPTION)
        ]
        assert ctx.get_power_usage(gpus=[1]) == [
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=_POWER_CONSUMPTION)
        ]
