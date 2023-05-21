from time import time_ns
from typing import Type, Optional, cast, List

import pynvml
from pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetClock, \
    nvmlDeviceGetTemperatureThreshold, nvmlDeviceGetClockInfo, nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature, \
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetFanSpeed_v2, nvmlDeviceGetFanSpeed, nvmlInit, nvmlShutdown

from gpulink.devices.base_device import BaseDevice
from gpulink.devices.gpu import Gpu, GpuSet
from gpulink.devices.nvml_defines import ClockType, ClockId, TemperatureSensorType, \
    TemperatureThreshold
from gpulink.devices.query import QueryResult, SimpleResult, MemInfo


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
            self._device_names.append(nvmlDeviceGetName(handle))

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
        try:
            nvmlInit()
            self._get_device_handles()
        except pynvml.nvml.NVMLError as e:
            raise RuntimeError("Cannot initialize NVML library - Is it installed?")

    def shutdown(self) -> None:
        nvmlShutdown()

    def get_gpus(self) -> GpuSet:
        gpus = []
        for id, name in zip(self._device_ids, self._device_names):
            gpus.append(Gpu(id, name))
        return GpuSet(gpus)

    def get_memory_info(self, gpus: Optional[List[int]] = None) -> List[MemInfo]:
        return cast(List[MemInfo], self._execute(nvmlDeviceGetMemoryInfo, MemInfo, gpus))

    def get_fan_speed(self, fan: Optional[int] = None, gpus: Optional[List[int]] = None) -> List[SimpleResult]:
        if fan is not None:
            return cast(List[SimpleResult], self._execute(nvmlDeviceGetFanSpeed_v2, SimpleResult, gpus, fan))
        else:
            return cast(List[SimpleResult], self._execute(nvmlDeviceGetFanSpeed, SimpleResult, gpus))

    def get_temperature(self, sensor_type: TemperatureSensorType, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        return cast(List[SimpleResult],
                    self._execute(nvmlDeviceGetTemperature, SimpleResult, gpus, sensor_type.value))

    def get_temperature_threshold(self, threshold: TemperatureThreshold, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        return cast(List[SimpleResult],
                    self._execute(nvmlDeviceGetTemperatureThreshold, SimpleResult, gpus, threshold.value))

    def get_clock(self, clock_type: ClockType, clock_id: ClockId = None, gpus: Optional[List[int]] = None) -> \
            List[SimpleResult]:
        if clock_id is not None:
            return cast(List[SimpleResult],
                        self._execute(nvmlDeviceGetClock, SimpleResult, gpus, clock_type.value, clock_id.value))
        else:
            return cast(List[SimpleResult],
                        self._execute(nvmlDeviceGetClockInfo, SimpleResult, gpus, clock_type.value))

    def get_power_usage(self, gpus: Optional[List[int]]) -> List[SimpleResult]:
        return cast(List[SimpleResult],
                    self._execute(nvmlDeviceGetPowerUsage, SimpleResult, gpus))
