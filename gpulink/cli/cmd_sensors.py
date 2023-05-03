from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Iterator

from tabulate import tabulate

from gpulink import DeviceCtx
from gpulink.cli.console import cls, set_cursor, get_spinner
from gpulink.cli.tools import start_in_background
from gpulink.consts import MB
from gpulink.devices.gpu import GpuSet
from gpulink.threading.stoppable_thread import StoppableThread
from gpulink.devices.nvml_defines import TemperatureSensorType, ClockType
from gpulink.devices.query import SimpleResult, MemInfo


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


@dataclass
class SensorStatus:
    """
    A container for storing several sensor status.
    """
    gpus: GpuSet
    memory: List[MemInfo]
    temperature: List[SimpleResult]
    fan_speed: List[SimpleResult]
    clock: Iterator

    def __str__(self):
        header = ["GPU", "Name", "Memory [MB]", "Temp [°C]", "Fan speed [%]", "Clock [MHz]"]
        table = [header]
        for data in zip(self.gpus, self.memory, self.temperature, self.fan_speed, self.clock):
            table.append([
                f"{data[0].id}",
                f"{data[0].name}",
                f"{int(data[1].used / MB)} / {int(data[1].total / MB)} ({(data[1].used / data[1].total) * 100:.1f}%)",
                f"{data[2].value}",
                f"{data[3].value}",
                f"Graph.: {data[4][0].value}\nMemory: {data[4][1].value}\n"
                f"SM: {data[4][2].value}\nVideo: {data[4][3].value} "
            ])
        return tabulate(table, headers='firstrow', tablefmt='fancy_grid')