from typing import Optional, List

from gpulink.devices.base_device import BaseDevice
from gpulink.devices.nvml_defines import TemperatureThreshold, ClockId, ClockType, \
    TemperatureSensorType
from gpulink.devices.query import SimpleResult, MemInfo
from gpulink.devices.gpu import Gpu, GpuSet

TEST_GB = int(1e9)
TEST_FAN_SPEED_PCT = 100
TEST_TEMP = 30
TEST_CLOCK = 100
TEST_POWER_CONSUMPTION = 30


class DeviceMock(BaseDevice):

    def setup(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_gpus(self) -> GpuSet:
        return GpuSet([Gpu(0, "GPU_0"), Gpu(1, "GPU_1")])

    def get_memory_info(self, gpus: Optional[List[int]] = None) -> List[MemInfo]:
        infos = [MemInfo(total=TEST_GB, used=TEST_GB // 2, free=TEST_GB // 2, gpu_idx=0, timestamp=0, gpu_name="GPU_0"),
                 MemInfo(total=TEST_GB, used=TEST_GB // 2, free=TEST_GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_1")]
        return infos if not gpus else [infos[i] for i in gpus]

    def get_fan_speed(self, fan=None, gpus: Optional[List[int]] = None) -> List[SimpleResult]:
        if not fan or fan == 0:
            speed = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_FAN_SPEED_PCT // 2),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_FAN_SPEED_PCT // 2)
            ]
        else:
            speed = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_FAN_SPEED_PCT // 4),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_FAN_SPEED_PCT // 4)
            ]
        return speed if not gpus else [speed[i] for i in gpus]

    def get_temperature(self, sensor_type: TemperatureSensorType, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        temps = [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_TEMP),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_TEMP)
        ]
        return temps if not gpus else [temps[i] for i in gpus]

    def get_temperature_threshold(self, threshold: TemperatureThreshold, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        if threshold == TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX:
            temps = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_TEMP // 2),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_TEMP // 2)
            ]
        else:
            temps = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_TEMP // 4),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_TEMP // 4)
            ]
        return temps if not gpus else [temps[i] for i in gpus]

    def get_clock(self, clock_type: ClockType, clock_id: ClockId = None, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:

        if not clock_id or clock_id == ClockId.CLOCK_ID_APP_CLOCK_DEFAULT:
            clock = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_CLOCK),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_CLOCK)
            ]
        else:
            clock = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_CLOCK // 2),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_CLOCK // 2)
            ]
        return clock if not gpus else [clock[i] for i in gpus]

    def get_power_usage(self, gpus: Optional[List[int]]) -> List[SimpleResult]:
        power = [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TEST_POWER_CONSUMPTION),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TEST_POWER_CONSUMPTION)
        ]
        return power if not gpus else [power[i] for i in gpus]
