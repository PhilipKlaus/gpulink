from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import List, Callable, Tuple, Union, Optional, Any

import numpy as np

from gpulink import DeviceCtx
from gpulink.devices.gpu import GpuSet
from gpulink.devices.nvml_defines import TemperatureSensorType, ClockType
from gpulink.devices.nvml_device import LocalNvmlGpu
from gpulink.devices.query import QueryResult
from gpulink.recording.gpu_recording import Recording, RecType
from gpulink.recording.timeseries import TimeSeries
from gpulink.threading.stoppable_thread import StoppableThread

EchoFunction = Optional[Callable[[], None]]


@dataclass
class _Recording:
    def __init__(self):
        self._timestamps = []
        self._data = []

    def add_record(self, timestamp, data):
        self._timestamps.append(timestamp)
        self._data.append(data)

    def to_timeseries(self) -> TimeSeries:
        return TimeSeries(
            timestamps=np.array(self._timestamps),
            data=np.array(self._data)
        )



class Recorder(StoppableThread):

    def __init__(
            self,
            cmd: Callable[[DeviceCtx], List[QueryResult]],
            res_filter: Callable[[QueryResult], Union[int, float, str]],
            ctx: DeviceCtx,
            rtype: RecType,
            runit: str,
            gpus: Optional[List[int]] = None,
            name: Optional[str] = None,
            echo_function: EchoFunction = None
    ):
        super().__init__()
        self._cmd = cmd
        self._filter = res_filter
        self._ctx = ctx
        self._gpus = gpus if gpus else ctx.gpus.ids
        self._rtype = rtype
        self._runit = runit
        self._name = name if name else "GPULink Recording"
        self._echo_function = echo_function
        self._recordings = [_Recording() for _ in self._gpus]

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(auto_join=True)

    def _get_record(self) -> Tuple[List, List[int]]:
        data = []
        timestamps = []
        for result in self._cmd(self._ctx):
            data.append(self._filter(result))
            timestamps.append(result.timestamp)
        return timestamps, data

    def _fetch_and_store(self):
        timestamps, data = self._get_record()
        for idx, record in enumerate(zip(timestamps, data)):
            self._recordings[idx].add_record(record[0], record[1])

    def run(self):
        old_time = 0
        while not self.should_stop:
            self._fetch_and_store()
            new_time = perf_counter()
            if self._echo_function and new_time - old_time > 0.10:
                self._echo_function()
                old_time = new_time

    def get_recording(self) -> Recording:
        return Recording(
            gpus=GpuSet([self._ctx.gpus[idx] for idx in self._gpus]),
            timeseries=[r.to_timeseries() for r in self._recordings],
            rtype=self._rtype,
            name=self._name,
            unit=self._runit)

    @classmethod
    def create_memory_recorder(cls, ctx: DeviceCtx, gpus: Optional[List[int]] = None, name: Optional[str] = None,
                               echo_function: EchoFunction = None):

        return cls(
            cmd=lambda c: c.get_memory_info(gpus),
            res_filter=lambda res: res.used,
            ctx=ctx,
            gpus=gpus,
            rtype=RecType.REC_TYPE_MEMORY,
            runit="Byte",
            name=name,
            echo_function=echo_function
        )

    @classmethod
    def create_temperature_recorder(cls, ctx: DeviceCtx, gpus: Optional[List[int]] = None, name: Optional[str] = None,
                                    echo_function: EchoFunction = None):

        return cls(
            cmd=lambda c: c.get_temperature(sensor_type=TemperatureSensorType.GPU, gpus=gpus),
            res_filter=lambda res: res.value,
            ctx=ctx,
            gpus=gpus,
            rtype=RecType.REC_TYPE_TEMPERATURE,
            runit="°C",
            name=name,
            echo_function=echo_function
        )

    @classmethod
    def create_fan_speed_recorder(cls, ctx: DeviceCtx, gpus: Optional[List[int]] = None, name: Optional[str] = None,
                                  echo_function: EchoFunction = None):

        return cls(
            cmd=lambda c: c.get_fan_speed(gpus=gpus),
            res_filter=lambda res: res.value,
            ctx=ctx,
            gpus=gpus,
            rtype=RecType.REC_TYPE_FAN_SPEED,
            runit="%",
            name=name,
            echo_function=echo_function
        )

    @classmethod
    def create_power_usage_recorder(cls, ctx: DeviceCtx, gpus: Optional[List[int]] = None, name: Optional[str] = None,
                                    echo_function: EchoFunction = None):

        return cls(
            cmd=lambda c: c.get_power_usage(gpus=gpus),
            res_filter=lambda res: res.value,
            ctx=ctx,
            gpus=gpus,
            rtype=RecType.REC_TYPE_POWER_USAGE,
            runit="W",
            name=name,
            echo_function=echo_function
        )

    @classmethod
    def create_clock_recorder(cls, ctx: DeviceCtx, clock_type: ClockType, gpus: Optional[List[int]] = None,
                              name: Optional[str] = None, echo_function: EchoFunction = None):

        clock_type_map = {
            ClockType.CLOCK_SM: RecType.REC_TYPE_CLOCK_SM,
            ClockType.CLOCK_MEM: RecType.REC_TYPE_CLOCK_MEM,
            ClockType.CLOCK_GRAPHICS: RecType.REC_TYPE_CLOCK_GRAPHICS,
            ClockType.CLOCK_VIDEO: RecType.REC_TYPE_CLOCK_VIDEO,
        }

        return cls(
            cmd=lambda c: c.get_clock(clock_type, gpus=gpus),
            res_filter=lambda res: res.value,
            ctx=ctx,
            gpus=gpus,
            rtype=clock_type_map[clock_type],
            runit="MHz",
            name=name,
            echo_function=echo_function
        )

    @classmethod
    def create_graphics_clock_recorder(cls, ctx: DeviceCtx, gpus: Optional[List[int]] = None,
                                       name: Optional[str] = None, echo_function: EchoFunction = None):
        return cls.create_clock_recorder(ctx, ClockType.CLOCK_GRAPHICS, gpus, name, echo_function)

    @classmethod
    def create_video_clock_recorder(cls, ctx: DeviceCtx, gpus: Optional[List[int]], name: Optional[str] = None,
                                    echo_function: EchoFunction = None):
        return cls.create_clock_recorder(ctx, ClockType.CLOCK_VIDEO, gpus, name, echo_function)

    @classmethod
    def create_sm_clock_recorder(cls, ctx: DeviceCtx, gpus: Optional[List[int]], name: Optional[str] = None,
                                 echo_function: EchoFunction = None):
        return cls.create_clock_recorder(ctx, ClockType.CLOCK_SM, gpus, name, echo_function)

    @classmethod
    def create_memory_clock_recorder(cls, ctx: DeviceCtx, gpus: Optional[List[int]], name: Optional[str] = None,
                                     echo_function: EchoFunction = None):
        return cls.create_clock_recorder(ctx, ClockType.CLOCK_MEM, gpus, name, echo_function)

    @classmethod
    def create_recorder(cls, ctx: DeviceCtx, rtype: RecType, gpus: Optional[List[int]] = None,
                        name: Optional[str] = None, echo_function: EchoFunction = None):
        if rtype == RecType.REC_TYPE_TEMPERATURE:
            return Recorder.create_temperature_recorder(ctx, gpus, name, echo_function)
        elif rtype == RecType.REC_TYPE_CLOCK_SM:
            return Recorder.create_sm_clock_recorder(ctx, gpus, name, echo_function)
        elif rtype == RecType.REC_TYPE_CLOCK_VIDEO:
            return Recorder.create_video_clock_recorder(ctx, gpus, name, echo_function)
        elif rtype == RecType.REC_TYPE_CLOCK_GRAPHICS:
            return Recorder.create_graphics_clock_recorder(ctx, gpus, name, echo_function)
        elif rtype == RecType.REC_TYPE_CLOCK_MEM:
            return Recorder.create_memory_clock_recorder(ctx, gpus, name, echo_function)
        elif rtype == RecType.REC_TYPE_FAN_SPEED:
            return Recorder.create_fan_speed_recorder(ctx, gpus, name, echo_function)
        elif rtype == RecType.REC_TYPE_MEMORY:
            return Recorder.create_memory_recorder(ctx, gpus, name, echo_function)
        elif rtype == RecType.REC_TYPE_POWER_USAGE:
            return Recorder.create_power_usage_recorder(ctx, gpus, name, echo_function)
        else:
            raise ValueError(f"Invalid RecType provided")


@dataclass
class RecWrapper:
    value: Any
    recording: Recording


def record(rtype: RecType, ctx_class=LocalNvmlGpu, gpus: Optional[List[int]] = None, name: str = None,
           echo_function: EchoFunction = None):
    """
    A decorator for recording GPU stats.
    :param rtype: Specifies the recorder type.
    :param ctx_class: The GPU device context (default: LocalNvmlGpu).
    :param gpus: A list of GPU ids to be recorded from.
    :param name: An optional name for the recording. If not provided __name__ of the decorated function is used.
    :param echo_function: An optional echo function which is called after recording a data frame.
    :return: Wrapped function.
    """

    def record_inner(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs) -> RecWrapper:
            with DeviceCtx(device=ctx_class) as ctx:
                rec_name = name if name else fn.__name__
                recorder = Recorder.create_recorder(ctx, rtype, gpus, rec_name, echo_function)
                with recorder:
                    ret_val = fn(*args, **kwargs)
                return RecWrapper(value=ret_val, recording=recorder.get_recording())

        return wrapped

    return record_inner
