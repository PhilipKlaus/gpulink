from typing import Optional, List

from gpulink.types import ClockType, ClockId, TemperatureThreshold, \
    TemperatureSensorType, MemInfo, SimpleResult, GpuSet


class BaseDevice:

    def setup(self) -> None:
        raise NotImplementedError()

    def shutdown(self) -> None:
        raise NotImplementedError()

    def get_gpus(self) -> GpuSet:
        raise NotImplementedError()

    def get_memory_info(self, gpus: Optional[List[int]]) -> List[MemInfo]:
        raise NotImplementedError()

    def get_fan_speed(self, gpus: Optional[List[int]], fan=None) -> List[SimpleResult]:
        raise NotImplementedError()

    def get_temperature(self, gpus: Optional[List[int]], sensor_type: TemperatureSensorType) -> \
            List[SimpleResult]:
        raise NotImplementedError()

    def get_temperature_threshold(self, gpus: Optional[List[int]], threshold: TemperatureThreshold) -> \
            List[SimpleResult]:
        raise NotImplementedError()

    def get_clock(self, gpus: Optional[List[int]], clock_type: ClockType, clock_id: ClockId = None) -> \
            List[SimpleResult]:
        raise NotImplementedError()

    def get_power_usage(self, gpus: Optional[List[int]]) -> List[SimpleResult]:
        raise NotImplementedError()
