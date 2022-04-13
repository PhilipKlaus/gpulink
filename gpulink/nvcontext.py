from time import time_ns
from typing import List, Type, cast, Optional

from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, \
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetFanSpeed_v2, nvmlDeviceGetFanSpeed, nvmlDeviceGetTemperature, \
    nvmlDeviceGetTemperatureThreshold, nvmlDeviceGetClock, nvmlDeviceGetClockInfo, nvmlDeviceGetPowerUsage

from gpulink.types import GPUMemInfo, GPUQueryResult, TemperatureThreshold, ClockId, \
    ClockType, GPUQuerySingleResult, TemperatureSensorType


class NVContext:
    """
    A context for executing nvml queries.
    """

    def __init__(self):
        self._device_handles = []
        self._device_names = []
        self._device_ids = []
        self._valid_ctx = False

    def __enter__(self):
        nvmlInit()
        self._valid_ctx = True
        self._get_device_handles()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        nvmlShutdown()
        self._valid_ctx = False

    def _get_device_handles(self):
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            self._device_handles.append(handle)
            self._device_names.append(nvmlDeviceGetName(handle).decode("utf-8"))
            self._device_ids.append(i)

    def _execute(self, query, type: Type, gpus: List[int], *args, **kwargs) -> List[GPUQueryResult]:
        if not self.valid_ctx:
            raise RuntimeError("Cannot execute query in an invalid NVContext")
        if not gpus or len(gpus) == 0:
            gpus = self._device_ids
            handles = self._device_handles
            gpu_names = self._device_names
        else:
            handles = [self._device_handles[gpu] for gpu in gpus]
            gpu_names = [self._device_names[gpu] for gpu in gpus]

        res = []
        for handle, name, idx in zip(handles, gpu_names, gpus):
            query_result = query(handle, *args, **kwargs)
            tmp = {"timestamp": time_ns(), "gpu_idx": idx, "gpu_name": name}

            keys = list(type.__annotations__)
            if len(keys) == 1:
                tmp[keys[0]] = query_result
            else:
                for key in type.__annotations__:
                    tmp[key] = getattr(query_result, key)
            res.append(type(**tmp))
        return res

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
        return self._device_ids

    @property
    def gpu_names(self) -> List[str]:
        """
        Queries the names of all active GPUs
        :return: The names of all active GPUs in a list
        """
        return self._device_names

    def get_memory_info(self, gpus: Optional[List[int]]) -> List[GPUMemInfo]:
        """
        Queries the memory information [Bytes] using nvmlDeviceGetMemoryInfo.
        :param gpus: A list of indices from GPU to be queried.
        :return: A list of GPUMemInfo.
        """
        return cast(List[GPUMemInfo], self._execute(nvmlDeviceGetMemoryInfo, GPUMemInfo, gpus))

    def get_fan_speed(self, gpus: Optional[List[int]], fan=None) -> List[GPUQuerySingleResult]:
        """
        Queries the fan speed [%] using nvmlDeviceGetFanSpeed_v2 and nvmlDeviceGetFanSpeed.
        :param gpus: A list of indices from GPU to be queried.
        :param fan: The index of the fan to be queried.
        :return: A list GPUQuerySingleResult.
        """
        if fan is not None:
            return cast(List[GPUQuerySingleResult], self._execute(nvmlDeviceGetFanSpeed_v2, GPUQuerySingleResult, gpus))
        else:
            return cast(List[GPUQuerySingleResult], self._execute(nvmlDeviceGetFanSpeed, GPUQuerySingleResult, gpus))

    def get_temperature(self, gpus: Optional[List[int]], sensor_type: TemperatureSensorType) -> \
            List[GPUQuerySingleResult]:
        """
        Queries the temperature [°C] using nvmlDeviceGetTemperature.
        :param gpus: A list of indices from GPU to be queried.
        :param sensor_type:The type of the actual temperature sensor to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        return cast(List[GPUQuerySingleResult],
                    self._execute(nvmlDeviceGetTemperature, GPUQuerySingleResult, gpus, sensor_type.value))

    def get_temperature_threshold(self, gpus: Optional[List[int]], threshold: TemperatureThreshold) -> \
            List[GPUQuerySingleResult]:
        """
        Queries the temperature threshold [°C] using nvmlDeviceGetTemperatureThreshold.
        :param gpus: A list of indices from GPU to be queried.
        :param threshold: The type of threshold to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        return cast(List[GPUQuerySingleResult],
                    self._execute(nvmlDeviceGetTemperatureThreshold, GPUQuerySingleResult, gpus, threshold.value))

    def get_clock(self, gpus: Optional[List[int]], clock_type: ClockType, clock_id: ClockId = None) -> \
            List[GPUQuerySingleResult]:
        """
        Queries the clock speed [MHz] using nvmlDeviceGetClock and nvmlDeviceGetClockInfo.
        :param gpus: A list of indices from GPU to be queried.
        :param clock_type: The type of clock to be queried.
        :param clock_id: The id of the clock to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        if clock_id is not None:
            return cast(List[GPUQuerySingleResult],
                        self._execute(nvmlDeviceGetClock, GPUQuerySingleResult, gpus, clock_type, clock_id.value))
        else:
            return cast(List[GPUQuerySingleResult],
                        self._execute(nvmlDeviceGetClockInfo, GPUQuerySingleResult, gpus, clock_type.value))

    def get_power_usage(self, gpus: Optional[List[int]]) -> List[GPUQuerySingleResult]:
        """
        Queries the power usage [mW] using nvmlDeviceGetPowerUsage.
        :param gpus: A list of indices from GPU to be queried.
        :return: A List of GPUQuerySingleResult.
        """
        return cast(List[GPUQuerySingleResult],
                    self._execute(nvmlDeviceGetPowerUsage, GPUQuerySingleResult, gpus))
