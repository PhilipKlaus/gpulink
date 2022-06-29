from typing import Optional, List

from gpulink.types import ClockType, ClockId, TemperatureThreshold, \
    TemperatureSensorType


class BaseDevice:

    def setup(self):
        raise NotImplementedError()

    def shutdown(self):
        raise NotImplementedError()

    def get_gpu_names(self):
        raise NotImplementedError()

    def get_gpu_ids(self):
        raise NotImplementedError()

    def get_memory_info(self, gpus: Optional[List[int]]):
        raise NotImplementedError()

    def get_fan_speed(self, gpus: Optional[List[int]], fan=None):
        raise NotImplementedError()

    def get_temperature(self, gpus: Optional[List[int]], sensor_type: TemperatureSensorType):
        raise NotImplementedError()

    def get_temperature_threshold(self, gpus: Optional[List[int]], threshold: TemperatureThreshold):
        raise NotImplementedError()

    def get_clock(self, gpus: Optional[List[int]], clock_type: ClockType, clock_id: ClockId = None):
        raise NotImplementedError()

    def get_power_usage(self, gpus: Optional[List[int]]):
        raise NotImplementedError()


