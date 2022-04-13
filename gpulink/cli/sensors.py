import os
import time

from colorama import Cursor

from gpulink import NVContext
from gpulink.cli.tools import get_spinner, busy_wait_for_interrupt
from gpulink.stoppable_thread import StoppableThread
from gpulink.types import TemperatureSensorType, ClockType, SensorStatus


def _set_cursor(x: int, y: int):
    return Cursor.POS(x, y)


class SensorWatcher(StoppableThread):
    def __init__(self, ctx: NVContext):
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

    def _update_view(self, spinner: str):
        print(f"{self.get_sensor_status()}\n[WATCHING] {spinner}{_set_cursor(1, 1)}", end="")

    def run(self) -> None:
        os.system("cls")
        spinner = get_spinner()
        while not self.should_stop:
            self._update_view(next(spinner))
            time.sleep(0.5)
        os.system("cls")


def sensors(args):
    """Print GPU sensor output"""

    with NVContext() as ctx:
        watcher = SensorWatcher(ctx)

        if args.watch:
            busy_wait_for_interrupt(watcher)
        else:
            print(watcher.get_sensor_status())
