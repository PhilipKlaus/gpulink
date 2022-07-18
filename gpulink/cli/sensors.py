import time

from gpulink import DeviceCtx
from gpulink.cli.console import cls, set_cursor, get_spinner
from gpulink.cli.tools import start_in_background
from gpulink.stoppable_thread import StoppableThread
from gpulink.gpu_types import TemperatureSensorType, ClockType, SensorStatus
from tests.misc import DeviceMock


class SensorWatcher(StoppableThread):
    def __init__(self, ctx: DeviceCtx):
        super().__init__()
        self._ctx = ctx

    def get_sensor_status(self) -> SensorStatus:
        gpus = self._ctx.gpus
        return SensorStatus(
            gpus=gpus,
            memory=self._ctx.get_memory_info(gpus=gpus.ids),
            temperature=self._ctx.get_temperature(TemperatureSensorType.GPU, gpus=gpus.ids),
            fan_speed=self._ctx.get_fan_speed(gpus=gpus.ids),
            clock=zip(
                self._ctx.get_clock(ClockType.CLOCK_GRAPHICS, gpus=gpus.ids),
                self._ctx.get_clock(ClockType.CLOCK_MEM, gpus=gpus.ids),
                self._ctx.get_clock(ClockType.CLOCK_SM, gpus=gpus.ids),
                self._ctx.get_clock(ClockType.CLOCK_VIDEO, gpus=gpus.ids),
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
