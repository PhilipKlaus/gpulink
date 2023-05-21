from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Iterator

import click
from tabulate import tabulate

from gpulink import DeviceCtx
from gpulink.cli.console import get_spinner, set_cursor
from gpulink.consts import MB, WATTS
from gpulink.devices.gpu import GpuSet
from gpulink.devices.nvml_defines import TemperatureSensorType, ClockType
from gpulink.devices.query import SimpleResult, MemInfo
from gpulink.threading.stoppable_thread import StoppableThread


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
            ),
            power_usage=self._ctx.get_power_usage(gpus.ids)
        )

    def run(self) -> None:
        click.clear()
        spinner = get_spinner()
        while not self.should_stop:
            click.echo(f"{self.get_sensor_status()}\nPress any key to abort...\n[WATCHING] ", nl=False)
            click.secho(f"{next(spinner)}{set_cursor(1, 1)}", nl=False, fg="green")
            time.sleep(0.1)
        click.clear()


@click.command(name="sensors")
@click.option('--watch', '-w', is_flag=True, help="Poll and print GPU sensor status.")
def sensors(watch: bool):
    """
    Fetch and print the GPU sensor status.
    :param watch: Set too bool if the sensor status should be polled and printed continuously.
    """
    with DeviceCtx() as ctx:
        watcher = SensorWatcher(ctx)
        if watch:
            watcher.start()
            click.pause("")
            watcher.stop(auto_join=True)
        click.echo(watcher.get_sensor_status())


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
    power_usage: List[SimpleResult]

    def __str__(self):
        header = ["GPU", "Name", "Memory [MB]", "Temp [°C]", "Fan speed [%]", "Clock [MHz]", "Power Usage [W]"]
        table = [header]
        for data in zip(self.gpus, self.memory, self.temperature, self.fan_speed, self.clock, self.power_usage):
            table.append([
                f"{data[0].id}",
                f"{data[0].name}",
                f"{int(data[1].used / MB)} / {int(data[1].total / MB)} ({(data[1].used / data[1].total) * 100:.1f}%)",
                f"{data[2].value}",
                f"{data[3].value}",
                f"Graph.: {data[4][0].value}\nMemory: {data[4][1].value}\n"
                f"SM: {data[4][2].value}\nVideo: {data[4][3].value}",
                f"{data[5].value / WATTS}"
            ])
        return tabulate(table, headers='firstrow', tablefmt='fancy_grid')
