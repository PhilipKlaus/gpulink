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

    def get_memory_info(self, gpus: Optional[List[int]] = None) -> List[MemInfo]:
        raise NotImplementedError()

    def get_fan_speed(self, fan: Optional[int] = None, gpus: Optional[List[int]] = None) -> List[SimpleResult]:
        raise NotImplementedError()

    def get_temperature(self, sensor_type: TemperatureSensorType, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        raise NotImplementedError()

    def get_temperature_threshold(self, threshold: TemperatureThreshold, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        raise NotImplementedError()

    def get_clock(self, clock_type: ClockType, clock_id: ClockId = None, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        raise NotImplementedError()

    def get_power_usage(self, gpus: Optional[List[int]]) -> List[SimpleResult]:
        raise NotImplementedError()
