import click

import gpulink
from gpulink.cli.cmd_record import record
from gpulink.cli.cmd_sensors import sensors


@click.group()
@click.version_option(version=gpulink.__version__)
def gpu_link():
    pass


gpu_link.add_command(sensors)
gpu_link.add_command(record)


def main():
    gpu_link(prog_name="GPU-Link: Monitor NVIDIA GPUs")


if __name__ == "__main__":
    main()
