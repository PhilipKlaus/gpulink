from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Callable, Optional, List

import click
from matplotlib import pyplot as plt

from gpulink import DeviceCtx, Plot, Recorder
from gpulink.cli.console import get_spinner, set_cursor
from gpulink.consts import MB
from gpulink.recording.gpu_recording import Recording


def _echo(spinner: cycle):
    click.echo("Press any key to abort...\n[RECORDING] ", nl=False)
    click.secho(f"{next(spinner)}{set_cursor(1, 1)}", nl=False, fg="green")


@dataclass
class _RecOptions:
    plot: bool
    autoscale: bool
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
    recording.plot_options.auto_scale = rec_options.autoscale
    graph = Plot(recording)
    graph.save(rec_options.output)


def _display_plot(recording: Recording, rec_options: _RecOptions):
    recording.plot_options.auto_scale = rec_options.autoscale
    p = Plot(recording)
    p.plot()


def _handle_record(rec_options: _RecOptions, factory_method: Callable, gpus: Optional[List[int]] = None):
    with DeviceCtx() as ctx:
        gpus = gpus if gpus else ctx.gpus.ids
        recorder = factory_method(ctx, gpus, echo_function=lambda: _echo(rec_options.spinner))
        with recorder:
            click.clear()
            click.pause(info="")
        click.clear()
        recording = recorder.get_recording()

        # If memory was recorded: convert the output to MB per default
        if factory_method == Recorder.create_memory_recorder:
            recording.convert(MB, "MB")

        click.echo(recording)

    if rec_options.output:
        _store_records(recording, rec_options)
    if rec_options.plot:
        _display_plot(recording, rec_options)


@click.group()
@click.option('--plot', '-p', is_flag=True, help="Displays a plot of the recorded GPU property over time.")
@click.option('--output', '-o', type=click.Path(), default=None, help="File path to store the GPU plot.")
@click.option('--no-autoscale', is_flag=True, help="Disable auto-scaling of the y axis in the plot.")
@click.pass_context
def record(ctx, plot: bool, output: str, no_autoscale: bool) -> None:
    """
    Record GPU properties.

    \f
    :param ctx: The Command context.
    :param plot: If true, a plot of the recorded GPU property is displayed.
    :param output: File path to store the GPU plot.
    :param no_autoscale: if true, auto-scaling of the y axis in the plot is disabled.
    :return: None
    """
    if output:
        output = Path(output)
        if not _check_output_file_type(output):
            ctx.exit(code=-1)

    ctx.obj = _RecOptions(
        plot=plot,
        autoscale=not no_autoscale,
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
