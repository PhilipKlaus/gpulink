import time

from gpulink import DeviceCtx
from gpulink.cli.console import cls, set_cursor, get_spinner
from gpulink.cli.tools import start_in_background
from gpulink.stoppable_thread import StoppableThread
from gpulink.types import TemperatureSensorType, ClockType, SensorStatus


class SensorWatcher(StoppableThread):
    def __init__(self, ctx: DeviceCtx):
        super().__init__()
        self._ctx = ctx

    def get_sensor_status(self) -> SensorStatus:
        gpus = self._ctx.gpus
        return SensorStatus(
            gpus=self._ctx.gpus,
            memory=self._ctx.get_memory_info(gpus),
            temperature=self._ctx.get_temperature(gpus, TemperatureSensorType.GPU),
            fan_speed=self._ctx.get_fan_speed(gpus),
            clock=zip(
                self._ctx.get_clock(gpus, ClockType.CLOCK_GRAPHICS),
                self._ctx.get_clock(gpus, ClockType.CLOCK_MEM),
                self._ctx.get_clock(gpus, ClockType.CLOCK_SM),
                self._ctx.get_clock(gpus, ClockType.CLOCK_VIDEO),
            )
        )

    def run(self) -> None:
        cls()
        spinner = get_spinner()
        while not self.should_stop:
            print(f"{self.get_sensor_status()}\n[WATCHING] {next(spinner)}{set_cursor(1, 1)}", end="")
            time.sleep(0.5)
        cls()


def sensors(args):
    """Print GPU sensor output"""

    with DeviceCtx() as ctx:
        watcher = SensorWatcher(ctx)

        if args.watch:
            start_in_background(watcher)
        else:
            print(watcher.get_sensor_status())
