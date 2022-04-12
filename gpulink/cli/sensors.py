import os
import time
from dataclasses import dataclass
from typing import List, Iterator

from colorama import Cursor
from tabulate import tabulate

from gpulink import NVContext
from gpulink.cli.tools import get_spinner, busy_wait_for_interrupt
from gpulink.consts import MB
from gpulink.stoppable_thread import StoppableThread
from gpulink.types import TemperatureSensorType, ClockType, GPUMemInfo, GPUQuerySingleResult


def _set_cursor(x: int, y: int):
    return Cursor.POS(x, y)


@dataclass
class SensorStatus:
    gpus: List[int]
    memory: List[GPUMemInfo]
    temperature: List[GPUQuerySingleResult]
    fan_speed: List[GPUQuerySingleResult]
    clock: Iterator

    def __str__(self):
        header = ["GPU", "Memory [MB]", "Temp [°C]", "Fan speed [%]", "Clock [MHz]"]
        table = [header]
        for data in zip(self.gpus, self.memory, self.temperature, self.fan_speed, self.clock):
            table.append([
                f"GPU[{data[0]}]",
                f"{int(data[1].used / MB)} / {int(data[1].total / MB)} ({(data[1].used / data[1].total) * 100:.1f}%)",
                f"{data[2].value}",
                f"{data[3].value}",
                f"Graph.: {data[4][0].value}\nMemory: {data[4][1].value}\nSM: {data[4][2].value}\nVideo: {data[4][3].value} "
            ])
        return tabulate(table, headers='firstrow', tablefmt='fancy_grid')


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
