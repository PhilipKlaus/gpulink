from typing import List, Optional

from gpulink.devices.base_device import BaseDevice
from gpulink.devices.nvml_device import LocalNvmlGpu
from gpulink.types import GPUMemInfo, TemperatureThreshold, ClockId, \
    ClockType, GPUQuerySingleResult, TemperatureSensorType


class NVContext:
    """
    A context for executing nvml queries.
    """

    def __init__(self, device_type: type[BaseDevice] = LocalNvmlGpu):
        self._valid_ctx = False
        self._device = device_type()

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
    def gpus(self) -> List[int]:
        """
        Queries the indices of all active GPUs
        :return: The indices of all active GPUs in a List
        """
        return self._device.get_gpu_ids()

    @property
    def gpu_names(self) -> List[str]:
        """
        Queries the names of all active GPUs
        :return: The names of all active GPUs in a list
        """
        return self._device.get_gpu_names()

    def get_memory_info(self, gpus: Optional[List[int]]) -> List[GPUMemInfo]:
        """
        Queries the memory information [Bytes] using nvmlDeviceGetMemoryInfo.
        :param gpus: A list of indices from GPU to be queried.
        :return: A list of GPUMemInfo.
        """
        return self._device.get_memory_info(gpus)

    def get_fan_speed(self, gpus: Optional[List[int]], fan=None) -> List[GPUQuerySingleResult]:
        """
        Queries the fan speed [%] using nvmlDeviceGetFanSpeed_v2 and nvmlDeviceGetFanSpeed.
        :param gpus: A list of indices from GPU to be queried.
        :param fan: The index of the fan to be queried.
        :return: A list GPUQuerySingleResult.
        """
        return self._device.get_fan_speed(gpus, fan)

    def get_temperature(self, gpus: Optional[List[int]], sensor_type: TemperatureSensorType) -> \
            List[GPUQuerySingleResult]:
        """
        Queries the temperature [°C] using nvmlDeviceGetTemperature.
        :param gpus: A list of indices from GPU to be queried.
        :param sensor_type:The type of the actual temperature sensor to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        return self._device.get_temperature(gpus, sensor_type)

    def get_temperature_threshold(self, gpus: Optional[List[int]], threshold: TemperatureThreshold) -> \
            List[GPUQuerySingleResult]:
        """
        Queries the temperature threshold [°C].
        :param gpus: A list of indices from GPU to be queried.
        :param threshold: The type of threshold to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        return self._device.get_temperature_threshold(gpus, threshold)

    def get_clock(self, gpus: Optional[List[int]], clock_type: ClockType, clock_id: ClockId = None) -> \
            List[GPUQuerySingleResult]:
        """
        Queries the clock speed [MHz].
        :param gpus: A list of indices from GPU to be queried.
        :param clock_type: The type of clock to be queried.
        :param clock_id: The id of the clock to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        return self._device.get_clock(gpus, clock_type, clock_id)

    def get_power_usage(self, gpus: Optional[List[int]]) -> List[GPUQuerySingleResult]:
        """
        Queries the power usage [mW].
        :param gpus: A list of indices from GPU to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        return self._device.get_power_usage(gpus)
