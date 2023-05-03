from typing import List, Callable, Tuple, Union, Optional

from gpulink import DeviceCtx
from gpulink.consts import MB
from gpulink.devices.gpu import GpuSet
from gpulink.devices.nvml_defines import TemperatureSensorType, ClockType
from gpulink.devices.query import QueryResult
from gpulink.plotting.plot_options import PlotOptions
from gpulink.recording.gpu_recording import Recording
from gpulink.recording.timeseries import TimeSeries
from gpulink.threading.stoppable_thread import StoppableThread


class Recorder(StoppableThread):

    def __init__(
            self,
            cmd: Callable[[DeviceCtx], List[QueryResult]],
            res_filter: Callable[[QueryResult], Union[int, float, str]],
            ctx: DeviceCtx,
            gpus: List[int],
            plot_options: Optional[PlotOptions] = None
    ):
        super().__init__()
        self._cmd = cmd
        self._filter = res_filter
        self._ctx = ctx
        self._gpus = gpus
        self._plot_options = plot_options

        self._timeseries = [TimeSeries() for _ in gpus]

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
        while not self.should_stop:
            self._fetch_and_store()

    def get_recording(self) -> Recording:
        return Recording(
            gpus=GpuSet([self._ctx.gpus[idx] for idx in self._gpus]),
            timeseries=self._timeseries,
            plot_options=self._plot_options
        )

    @classmethod
    def create_memory_recorder(cls, ctx: DeviceCtx, gpus: List[int], plot_options: Optional[PlotOptions] = None):
        default_options = PlotOptions(
            plot_name="Memory usage",
            y_axis_unit="MB",
            y_axis_label="Memory usage",
            y_axis_divider=MB,
            y_axis_range=(0, max([mem.total for mem in ctx.get_memory_info()])),
            auto_scale=False
        )
        if plot_options:
            default_options.patch(plot_options)
        return Recorder(
            cmd=lambda c: c.get_memory_info(gpus),
            res_filter=lambda res: res.used,
            ctx=ctx,
            gpus=gpus,
            plot_options=default_options
        )

    @classmethod
    def create_temperature_recorder(cls, ctx: DeviceCtx, gpus: List[int], plot_options: Optional[PlotOptions] = None):
        default_options = PlotOptions(
            plot_name="Temperature",
            y_axis_unit="°C",
            y_axis_label="Temperature",
            y_axis_divider=1,
            y_axis_range=None,
            auto_scale=True
        )
        if plot_options:
            default_options.patch(plot_options)
        return Recorder(
            cmd=lambda c: c.get_temperature(sensor_type=TemperatureSensorType.GPU, gpus=gpus),
            res_filter=lambda res: res.value,
            ctx=ctx,
            gpus=gpus,
            plot_options=default_options
        )

    @classmethod
    def create_fan_speed_recorder(cls, ctx: DeviceCtx, gpus: List[int], plot_options: Optional[PlotOptions] = None):
        default_options = PlotOptions(
            plot_name="Fan speed",
            y_axis_unit="%",
            y_axis_label="Fan speed",
            y_axis_divider=1,
            y_axis_range=(0, 100),
            auto_scale=False
        )
        if plot_options:
            default_options.patch(plot_options)
        return Recorder(
            cmd=lambda c: c.get_fan_speed(gpus=gpus),
            res_filter=lambda res: res.value,
            ctx=ctx,
            gpus=gpus,
            plot_options=default_options
        )

    @classmethod
    def create_clock_recorder(cls, ctx: DeviceCtx, gpus: List[int], clock_type: ClockType,
                              plot_options: Optional[PlotOptions] = None):
        default_options = PlotOptions(
            y_axis_unit="MHz",
            y_axis_divider=1,
            y_axis_range=None,
            auto_scale=True
        )
        if plot_options:
            default_options.patch(plot_options)
        return Recorder(
            cmd=lambda c: c.get_clock(clock_type, gpus=gpus),
            res_filter=lambda res: res.value,
            ctx=ctx,
            gpus=gpus,
            plot_options=default_options
        )

    @classmethod
    def create_graphics_clock_recorder(cls, ctx: DeviceCtx, gpus: List[int],
                                       plot_options: Optional[PlotOptions] = None):
        default_options = PlotOptions(plot_name="Graphics Clock", y_axis_label="Graphics Clock")
        if plot_options:
            default_options.patch(plot_options)
        return cls.create_clock_recorder(
            ctx, gpus, ClockType.CLOCK_GRAPHICS, default_options
        )
