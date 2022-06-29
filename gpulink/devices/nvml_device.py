from time import time_ns
from typing import Type, Optional, cast

from pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetClock, \
    nvmlDeviceGetTemperatureThreshold, nvmlDeviceGetClockInfo, nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature

from gpulink.devices.base_device import BaseDevice
from gpulink.types import GPUQueryResult, GPUQuerySingleResult, ClockType, ClockId


class List:
    pass


class LocalNvmlGpu(BaseDevice):

    def __init__(self):
        self._device_handles = []
        self._device_names = []
        self._device_ids = []

    def _get_device_handles(self):
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            self._device_handles.append(handle)
            self._device_names.append(nvmlDeviceGetName(handle).decode("utf-8"))
            self._device_ids.append(i)

    def _execute(self, query, type: Type, gpus: List[int], *args, **kwargs) -> List[GPUQueryResult]:
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

    def setup(self):
        nvmlInit()
        self._get_device_handles()

    def shutdown(self):
        nvmlShutdown()

    def get_gpu_ids(self) -> List[int]:
        return self._device_ids

    def get_gpu_names(self) -> List[str]:
        return self._device_names

    def get_memory_info(self, gpus: Optional[List[int]]) -> List[GPUMemInfo]:
        return cast(List[GPUMemInfo], self._execute(nvmlDeviceGetMemoryInfo, GPUMemInfo, gpus))

    def get_fan_speed(self, gpus: Optional[List[int]], fan=None) -> List[GPUQuerySingleResult]:
        if fan is not None:
            return cast(List[GPUQuerySingleResult], self._execute(nvmlDeviceGetFanSpeed_v2, GPUQuerySingleResult, gpus))
        else:
            return cast(List[GPUQuerySingleResult], self._execute(nvmlDeviceGetFanSpeed, GPUQuerySingleResult, gpus))

    def get_temperature(self, gpus: Optional[List[int]], sensor_type: TemperatureSensorType) -> \
            List[GPUQuerySingleResult]:
        return cast(List[GPUQuerySingleResult],
                    self._execute(nvmlDeviceGetTemperature, GPUQuerySingleResult, gpus, sensor_type.value))

    def get_temperature_threshold(self, gpus: Optional[List[int]], threshold: TemperatureThreshold) -> \
            List[GPUQuerySingleResult]:
        return cast(List[GPUQuerySingleResult],
                    self._execute(nvmlDeviceGetTemperatureThreshold, GPUQuerySingleResult, gpus, threshold.value))

    def get_clock(self, gpus: Optional[List[int]], clock_type: ClockType, clock_id: ClockId = None) -> \
            List[GPUQuerySingleResult]:
        if clock_id is not None:
            return cast(List[GPUQuerySingleResult],
                        self._execute(nvmlDeviceGetClock, GPUQuerySingleResult, gpus, clock_type, clock_id.value))
        else:
            return cast(List[GPUQuerySingleResult],
                        self._execute(nvmlDeviceGetClockInfo, GPUQuerySingleResult, gpus, clock_type.value))

    def get_power_usage(self, gpus: Optional[List[int]]):
        return cast(List[GPUQuerySingleResult],
                    self._execute(nvmlDeviceGetPowerUsage, GPUQuerySingleResult, gpus))