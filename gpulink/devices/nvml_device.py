from time import time_ns
from typing import Type, Optional, cast, List

from pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetClock, \
    nvmlDeviceGetTemperatureThreshold, nvmlDeviceGetClockInfo, nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature, \
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetFanSpeed_v2, nvmlDeviceGetFanSpeed, nvmlInit, nvmlShutdown

from gpulink.devices.base_device import BaseDevice
from gpulink.types import QueryResult, SimpleResult, ClockType, ClockId, MemInfo, TemperatureSensorType, \
    TemperatureThreshold


class LocalNvmlGpu(BaseDevice):

    def __init__(self):
        self._device_handles = []
        self._device_names = []
        self._device_ids = []

    def _get_device_handles(self):
        self._device_ids = [i for i in range(nvmlDeviceGetCount())]
        for dev in self._device_ids:
            handle = nvmlDeviceGetHandleByIndex(dev)
            self._device_handles.append(handle)
            self._device_names.append(nvmlDeviceGetName(handle).decode("utf-8"))

    def _execute(self, query, type: Type, gpus: List[int], *args, **kwargs) -> List[QueryResult]:
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

    def setup(self) -> None:
        nvmlInit()
        self._get_device_handles()

    def shutdown(self) -> None:
        nvmlShutdown()

    def get_gpu_ids(self) -> List[int]:
        return self._device_ids

    def get_gpu_names(self) -> List[str]:
        return self._device_names

    def get_memory_info(self, gpus: Optional[List[int]]) -> List[MemInfo]:
        return cast(List[MemInfo], self._execute(nvmlDeviceGetMemoryInfo, MemInfo, gpus))

    def get_fan_speed(self, gpus: Optional[List[int]], fan=None) -> List[SimpleResult]:
        if fan is not None:
            return cast(List[SimpleResult], self._execute(nvmlDeviceGetFanSpeed_v2, SimpleResult, gpus))
        else:
            return cast(List[SimpleResult], self._execute(nvmlDeviceGetFanSpeed, SimpleResult, gpus))

    def get_temperature(self, gpus: Optional[List[int]], sensor_type: TemperatureSensorType) -> \
            List[SimpleResult]:
        return cast(List[SimpleResult],
                    self._execute(nvmlDeviceGetTemperature, SimpleResult, gpus, sensor_type.value))

    def get_temperature_threshold(self, gpus: Optional[List[int]], threshold: TemperatureThreshold) -> \
            List[SimpleResult]:
        return cast(List[SimpleResult],
                    self._execute(nvmlDeviceGetTemperatureThreshold, SimpleResult, gpus, threshold.value))

    def get_clock(self, gpus: Optional[List[int]], clock_type: ClockType, clock_id: ClockId = None) -> \
            List[SimpleResult]:
        if clock_id is not None:
            return cast(List[SimpleResult],
                        self._execute(nvmlDeviceGetClock, SimpleResult, gpus, clock_type, clock_id.value))
        else:
            return cast(List[SimpleResult],
                        self._execute(nvmlDeviceGetClockInfo, SimpleResult, gpus, clock_type.value))

    def get_power_usage(self, gpus: Optional[List[int]]) -> List[SimpleResult]:
        return cast(List[SimpleResult],
                    self._execute(nvmlDeviceGetPowerUsage, SimpleResult, gpus))
