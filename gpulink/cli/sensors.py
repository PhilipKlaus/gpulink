import os
import time

from colorama import Cursor
from tabulate import tabulate

from gpulink import NVContext
from gpulink.cli.tools import get_spinner, busy_wait_for_interrupt
from gpulink.consts import MB
from gpulink.stoppable_thread import StoppableThread
from gpulink.types import TemperatureSensorType, ClockType


class SensorWatcher(StoppableThread):
    def __init__(self, ctx: NVContext):
        super().__init__()
        self._ctx = ctx

    def get_status(self):
        header = ["GPU", "Memory [MB]", "Temp [°C]", "Fan speed [%]", "Clock [MHz]"]
        table = [header]

        gpus = self._ctx.gpus
        mem = self._ctx.get_memory_info(gpus)
        tmp = self._ctx.get_temperature(gpus, TemperatureSensorType.GPU)
        fan = self._ctx.get_fan_speed(gpus)
        clock = zip(
            self._ctx.get_clock(gpus, ClockType.CLOCK_GRAPHICS),
            self._ctx.get_clock(gpus, ClockType.CLOCK_MEM),
            self._ctx.get_clock(gpus, ClockType.CLOCK_SM),
            self._ctx.get_clock(gpus, ClockType.CLOCK_VIDEO),
        )

        for data in zip(gpus, mem, tmp, fan, clock):
            table.append([
                f"GPU[{data[0]}]",
                f"{int(data[1].used / MB)} / {int(data[1].total / MB)} ({(data[1].used / data[1].total) * 100:.1f}%)",
                f"{data[2].value}",
                f"{data[3].value}",
                f"Graph.: {data[4][0].value}\nMemory: {data[4][1].value}\nSM: {data[4][2].value}\nVideo: {data[4][3].value} "
            ])
        return tabulate(table, headers='firstrow', tablefmt='fancy_grid')

    def run(self) -> None:
        os.system("cls")
        pos = lambda y, x: Cursor.POS(x, y)
        spinner = get_spinner()
        while not self.should_stop:
            print(f"{self.get_status()}\n[WATCHING] {next(spinner)}{pos(1, 1)}", end="")
            time.sleep(0.5)
        os.system("cls")


def sensors(args):
    """Print GPU sensor output"""

    with NVContext() as ctx:
        watcher = SensorWatcher(ctx)

        if args.watch:
            busy_wait_for_interrupt(watcher)
        else:
            print(watcher.get_status())
