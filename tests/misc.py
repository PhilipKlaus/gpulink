from typing import Optional, List

from gpulink.devices.base_device import BaseDevice
from gpulink.types import GpuSet, Gpu, MemInfo, SimpleResult, TemperatureThreshold, ClockId, ClockType, \
    TemperatureSensorType

GB = int(1e9)
FAN_SPEED_PCT = 100
TMP = 30
CLOCK = 100
POWER_CONSUMPTION = 30


class DeviceMock(BaseDevice):

    def setup(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_gpus(self) -> GpuSet:
        return GpuSet([Gpu(0, "GPU_0"), Gpu(1, "GPU_1")])

    def get_memory_info(self, gpus: Optional[List[int]] = None) -> List[MemInfo]:
        infos = [MemInfo(total=GB, used=GB // 2, free=GB // 2, gpu_idx=0, timestamp=0, gpu_name="GPU_0"),
                 MemInfo(total=GB, used=GB // 2, free=GB // 2, gpu_idx=1, timestamp=0, gpu_name="GPU_1")]
        return infos if not gpus else [infos[i] for i in gpus]

    def get_fan_speed(self, fan=None, gpus: Optional[List[int]] = None) -> List[SimpleResult]:
        if not fan or fan == 0:
            speed = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=FAN_SPEED_PCT // 2),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=FAN_SPEED_PCT // 2)
            ]
        else:
            speed = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=FAN_SPEED_PCT // 4),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=FAN_SPEED_PCT // 4)
            ]
        return speed if not gpus else [speed[i] for i in gpus]

    def get_temperature(self, sensor_type: TemperatureSensorType, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        temps = [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TMP),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TMP)
        ]
        return temps if not gpus else [temps[i] for i in gpus]

    def get_temperature_threshold(self, threshold: TemperatureThreshold, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        if threshold == TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX:
            temps = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TMP // 2),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TMP // 2)
            ]
        else:
            temps = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=TMP // 4),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=TMP // 4)
            ]
        return temps if not gpus else [temps[i] for i in gpus]

    def get_clock(self, clock_type: ClockType, clock_id: ClockId = None, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:

        if not clock_id or clock_id == ClockId.CLOCK_ID_APP_CLOCK_DEFAULT:
            clock = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=CLOCK),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=CLOCK)
            ]
        else:
            clock = [
                SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=CLOCK // 2),
                SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=CLOCK // 2)
            ]
        return clock if not gpus else [clock[i] for i in gpus]

    def get_power_usage(self, gpus: Optional[List[int]]) -> List[SimpleResult]:
        power = [
            SimpleResult(gpu_idx=0, timestamp=0, gpu_name="GPU_0", value=POWER_CONSUMPTION),
            SimpleResult(gpu_idx=1, timestamp=0, gpu_name="GPU_1", value=POWER_CONSUMPTION)
        ]
        return power if not gpus else [power[i] for i in gpus]
