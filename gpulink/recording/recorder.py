from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import List, Callable, Tuple, Union, Optional, Any

from gpulink import DeviceCtx
from gpulink.consts import RecType
from gpulink.devices.gpu import GpuSet
from gpulink.devices.nvml_defines import TemperatureSensorType, ClockType
from gpulink.devices.nvml_device import LocalNvmlGpu
from gpulink.devices.query import QueryResult
from gpulink.recording.gpu_recording import Recording
from gpulink.recording.timeseries import TimeSeries
from gpulink.threading.stoppable_thread import StoppableThread

EchoFunction = Optional[Callable[[], None]]


class Recorder(StoppableThread):

    def __init__(
            self,
            cmd: Callable[[DeviceCtx], List[QueryResult]],
            res_filter: Callable[[QueryResult], Union[int, float, str]],
            ctx: DeviceCtx,
            gpus: List[int],
            rec_type: RecType,
            rec_name: Optional[str] = None,
            echo_function: EchoFunction = None
    ):
        super().__init__()
        self._cmd = cmd
        self._filter = res_filter
        self._ctx = ctx
        self._gpus = gpus
        self._rec_type = rec_type
        self._rec_name = rec_name if rec_name else "Recording"
        self._echo_function = echo_function
        self._timeseries = [TimeSeries() for _ in self._gpus]

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
            self._timeseries[idx].add_record(record[0], record[1])

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
            timeseries=self._timeseries,
            rec_type=self._rec_type,
            rec_name=self._rec_name)

    @classmethod
    def create_memory_recorder(cls, ctx: DeviceCtx, gpus: List[int], name: Optional[str] = None,
                               echo_function: EchoFunction = None):

        return cls(
            cmd=lambda c: c.get_memory_info(gpus),
            res_filter=lambda res: res.used,
            ctx=ctx,
            gpus=gpus,
            rec_type=RecType.REC_TYPE_MEMORY,
            rec_name=name,
            echo_function=echo_function
        )

    @classmethod
    def create_temperature_recorder(cls, ctx: DeviceCtx, gpus: List[int], name: Optional[str] = None,
                                    echo_function: EchoFunction = None):

        return cls(
            cmd=lambda c: c.get_temperature(sensor_type=TemperatureSensorType.GPU, gpus=gpus),
            res_filter=lambda res: res.value,
            ctx=ctx,
            gpus=gpus,
            rec_type=RecType.REC_TYPE_TEMPERATURE,
            rec_name=name,
            echo_function=echo_function
        )

    @classmethod
    def create_fan_speed_recorder(cls, ctx: DeviceCtx, gpus: List[int], name: Optional[str] = None,
                                  echo_function: EchoFunction = None):

        return cls(
            cmd=lambda c: c.get_fan_speed(gpus=gpus),
            res_filter=lambda res: res.value,
            ctx=ctx,
            gpus=gpus,
            rec_type=RecType.REC_TYPE_FAN_SPEED,
            rec_name=name,
            echo_function=echo_function
        )

    @classmethod
    def create_power_usage_recorder(cls, ctx: DeviceCtx, gpus: List[int], name: Optional[str] = None,
                                    echo_function: EchoFunction = None):

        return cls(
            cmd=lambda c: c.get_power_usage(gpus=gpus),
            res_filter=lambda res: res.value,
            ctx=ctx,
            gpus=gpus,
            rec_type=RecType.REC_TYPE_POWER_USAGE,
            rec_name=name,
            echo_function=echo_function
        )

    @classmethod
    def create_clock_recorder(cls, ctx: DeviceCtx, gpus: List[int], clock_type: ClockType, name: Optional[str] = None,
                              echo_function: EchoFunction = None):

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
            rec_type=clock_type_map[clock_type],
            rec_name=name,
            echo_function=echo_function
        )

    @classmethod
    def create_graphics_clock_recorder(cls, ctx: DeviceCtx, gpus: List[int], name: Optional[str] = None,
                                       echo_function: EchoFunction = None):
        return cls.create_clock_recorder(ctx, gpus, ClockType.CLOCK_GRAPHICS, name, echo_function)

    @classmethod
    def create_video_clock_recorder(cls, ctx: DeviceCtx, gpus: List[int], name: Optional[str] = None,
                                    echo_function: EchoFunction = None):
        return cls.create_clock_recorder(ctx, gpus, ClockType.CLOCK_VIDEO, name, echo_function)

    @classmethod
    def create_sm_clock_recorder(cls, ctx: DeviceCtx, gpus: List[int], name: Optional[str] = None,
                                 echo_function: EchoFunction = None):
        return cls.create_clock_recorder(ctx, gpus, ClockType.CLOCK_SM, name, echo_function)

    @classmethod
    def create_memory_clock_recorder(cls, ctx: DeviceCtx, gpus: List[int], name: Optional[str] = None,
                                     echo_function: EchoFunction = None):
        return cls.create_clock_recorder(ctx, gpus, ClockType.CLOCK_MEM, name, echo_function)


@dataclass
class RecWrapper:
    value: Any
    recording: Recording


def record(ctx_class=LocalNvmlGpu, factory=Callable[[Any], Recorder], gpus: List[int] = None, name: str = None,
           **kwargs_record):
    """
    A decorator for recording GPU stats.
    :param ctx_class: The GPU device context (default: LocalNvmlGpu).
    :param factory: The factory method used for instantiating the recorder.
    :param gpus: A list of GPU ids to be recorded from.
    :param name: An optional name for the recording. If not provided __name__ of the decorated function is used.
    :param kwargs_record: Additional keyword argument for  the factory method (e.g. plot_options)
    :return: Wrapped function.
    """

    def record_inner(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs) -> RecWrapper:
            with DeviceCtx(device=ctx_class) as ctx:
                rec_name = name if name else fn.__name__
                used_gpus = ctx.gpus.ids if not gpus else gpus
                recorder = factory(ctx, used_gpus, rec_name, **kwargs_record)
                with recorder:
                    ret_val = fn(*args, **kwargs)
                return RecWrapper(value=ret_val, recording=recorder.get_recording())

        return wrapped

    return record_inner
