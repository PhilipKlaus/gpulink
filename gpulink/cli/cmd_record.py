from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional, List

import click
from matplotlib import pyplot as plt

from gpulink import DeviceCtx, Plot, Recorder
from gpulink.cli.console import get_spinner, set_cursor
from gpulink.consts import MB, WATTS
from gpulink.recording.gpu_recording import Recording


class Callback:
    def __init__(self, spinner: cycle):
        self._spinner = spinner
        self._last_ts = 0

    def echo(self, _, __):
        def _echo(spinner: cycle):
            click.echo("Press any key to abort...\n[RECORDING] ", nl=False)
            click.secho(f"{next(spinner)}{set_cursor(1, 1)}", nl=False, fg="green")

        ts = perf_counter()
        if ts - self._last_ts > 0.10:
            _echo(self._spinner)
            self._last_ts = ts


# Global variable to store the actual recording callback
_callback: Optional[Callback] = None


@dataclass
class _RecOptions:
    plot: bool
    output: Optional[Path] = None
    spinner = get_spinner()


def _check_output_file_type(output_path: Path) -> bool:
    supported_file_types = plt.gcf().canvas.get_supported_filetypes()
    # Necessary to ensure that implicitly created figure is deleted
    plt.clf()
    plt.cla()
    plt.close()
    if output_path and output_path.suffix[1:] not in supported_file_types:
        click.secho(f"Output format '{output_path.suffix}' not supported", fg="red")
        return False
    return True


def _store_records(recording: Recording, rec_options: _RecOptions):
    graph = Plot(recording)
    graph.save(rec_options.output)


def _display_plot(recording: Recording):
    p = Plot(recording)
    p.plot()


def _handle_record(rec_options: _RecOptions, factory_method: Callable, gpus: Optional[List[int]] = None):
    global _callback
    with DeviceCtx() as ctx:
        gpus = gpus if gpus else ctx.gpus.ids
        _callback = Callback(rec_options.spinner)
        recorder = factory_method(ctx, gpus, callback=_callback.echo)
        with recorder:
            click.clear()
            click.pause(info="")
        click.clear()
        recording = recorder.get_recording()

        # If memory was recorded: convert the output to MB per default
        if factory_method == Recorder.create_memory_recorder:
            recording.convert(MB, "MB")

        # If power-consumption was recorded: convert the output to W per default
        if factory_method == Recorder.create_power_usage_recorder:
            recording.convert(WATTS, "W")

        click.echo(recording)

    if rec_options.output:
        _store_records(recording, rec_options)
    if rec_options.plot:
        _display_plot(recording)


@click.group()
@click.option('--plot', '-p', is_flag=True, help="Displays a plot of the recorded GPU property over time.")
@click.option('--output', '-o', type=click.Path(), default=None, help="File path to store the GPU plot.")
@click.pass_context
def record(ctx, plot: bool, output: str) -> None:
    """
    Record GPU properties.

    \f
    :param ctx: The Command context.
    :param plot: If true, a plot of the recorded GPU property is displayed.
    :param output: File path to store the GPU plot.
    :return: None
    """
    if output:
        output = Path(output)
        if not _check_output_file_type(output):
            ctx.exit(code=-1)

    ctx.obj = _RecOptions(
        plot=plot,
        output=output
    )


@record.command()
@click.pass_obj
def memory(rec_options: _RecOptions) -> None:
    """
    Record GPU memory usage.
    \f
    :return: None
    """
    _handle_record(rec_options, Recorder.create_memory_recorder)


@record.command()
@click.pass_obj
def temp(rec_options: _RecOptions) -> None:
    """
    Record GPU temperature.
    \f
    :return: None
    """
    _handle_record(rec_options, Recorder.create_temperature_recorder)


@record.command()
@click.pass_obj
def fan_speed(rec_options: _RecOptions) -> None:
    """
    Record GPU fan speed.
    \f
    :return: None
    """
    _handle_record(rec_options, Recorder.create_fan_speed_recorder)


@record.command()
@click.pass_obj
def power_usage(rec_options: _RecOptions) -> None:
    """
    Record GPU power usage.
    \f
    :return: None
    """
    _handle_record(rec_options, Recorder.create_power_usage_recorder)


@record.group()
@click.pass_obj
def clock(rec_options: _RecOptions) -> None:
    """
    Record a GPU clock (sm,graphics,memory or video).

    \f
    :param rec_options: The recording options.
    :return: None
    """
    pass


@clock.command()
@click.pass_obj
def graphics(rec_options: _RecOptions) -> None:
    """
    Record the GPU Graphics clock.

    \f
    :param rec_options: The recording options.
    :return: None
    """
    _handle_record(rec_options, Recorder.create_graphics_clock_recorder)


@clock.command()
@click.pass_obj
def sm(rec_options: _RecOptions) -> None:
    """
    Record the GPU Graphics clock.

    \f
    :param rec_options: The recording options.
    :return: None
    """
    _handle_record(rec_options, Recorder.create_sm_clock_recorder)


@clock.command()
@click.pass_obj
def video(rec_options: _RecOptions) -> None:
    """
    Record the GPU Graphics clock.

    \f
    :param rec_options: The recording options.
    :return: None
    """
    _handle_record(rec_options, Recorder.create_video_clock_recorder)


@clock.command()
@click.pass_obj
def memory(rec_options: _RecOptions) -> None:
    """
    Record the GPU Graphics clock.

    \f
    :param rec_options: The recording options.
    :return: None
    """
    _handle_record(rec_options, Recorder.create_memory_clock_recorder)
