from functools import wraps
from typing import List, Optional, Type

from gpulink.devices.base_device import BaseDevice
from gpulink.devices.nvml_device import LocalNvmlGpu
from gpulink.types import MemInfo, TemperatureThreshold, ClockId, \
    ClockType, SimpleResult, TemperatureSensorType, GpuSet


def ctx_guard(fn):
    @wraps(fn)
    def guard(ref, *args, **kwargs):
        if not ref.valid_ctx:
            raise RuntimeError("Cannot execute query in an invalid NVContext")
        return fn(ref, *args, **kwargs)

    return guard


class DeviceCtx:
    """
    A context for executing nvml queries.
    """

    def __init__(self, device: Type[BaseDevice] = LocalNvmlGpu):
        self._valid_ctx = False
        self._device = device()

    def __enter__(self):
        self._device.setup()
        self._valid_ctx = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._device.shutdown()
        self._valid_ctx = False

    @property
    def valid_ctx(self):
        """
        Queries if the actual NVContext is valid.
        :return: True if the actual NVContext is valid, else False
        """
        return self._valid_ctx

    @property
    @ctx_guard
    def gpus(self) -> GpuSet:
        """
        Queries the indices of all active GPUs
        :return: The indices of all active GPUs in a List
        """
        return self._device.get_gpus()

    @ctx_guard
    def get_memory_info(self, gpus: Optional[List[int]] = None) -> List[MemInfo]:
        """
        Queries the memory information [Bytes] using nvmlDeviceGetMemoryInfo.
        :param gpus: A list of indices from GPU to be queried.
        :return: A list of GPUMemInfo.
        """
        return self._device.get_memory_info(gpus)

    @ctx_guard
    def get_fan_speed(self, fan: Optional[int] = None, gpus: Optional[List[int]] = None) -> List[SimpleResult]:
        """
        Queries the fan speed [%] using nvmlDeviceGetFanSpeed_v2 and nvmlDeviceGetFanSpeed.
        :param fan: The index of the fan to be queried.
        :param gpus: An optional list of GPU indices to be queried.
        :return: A list GPUQuerySingleResult.
        """
        return self._device.get_fan_speed(fan, gpus)

    @ctx_guard
    def get_temperature(self, sensor_type: TemperatureSensorType, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        """
        Queries the temperature [°C] using nvmlDeviceGetTemperature.
        :param sensor_type:The type of the actual temperature sensor to be queried.
        :param gpus: An optional list of GPU indices to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        return self._device.get_temperature(sensor_type, gpus)

    @ctx_guard
    def get_temperature_threshold(self, threshold: TemperatureThreshold, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        """
        Queries the temperature threshold [°C].
        :param gpus: A list of indices from GPU to be queried.
        :param threshold: The type of threshold to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        return self._device.get_temperature_threshold(threshold, gpus)

    @ctx_guard
    def get_clock(self, clock_type: ClockType, clock_id: ClockId = None, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        """
        Queries the clock speed [MHz].
        :param gpus: A list of indices from GPU to be queried.
        :param clock_type: The type of clock to be queried.
        :param clock_id: The id of the clock to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        return self._device.get_clock(clock_type, clock_id, gpus)

    @ctx_guard
    def get_power_usage(self, gpus: Optional[List[int]]) -> List[SimpleResult]:
        """
        Queries the power usage [mW].
        :param gpus: A list of indices from GPU to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        return self._device.get_power_usage(gpus)
