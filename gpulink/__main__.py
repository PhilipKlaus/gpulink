import argparse
from pathlib import Path

import colorama

from gpulink.cli.record import record
from gpulink.cli.sensors import sensors


def main():
    colorama.init()

    parser = argparse.ArgumentParser(description="GPU-Link: Monitor NVIDIA GPU status")
    subparser = parser.add_subparsers(dest="subcommand")

    parser_sensors = subparser.add_parser("sensors", description="Print GPU sensor output")
    parser_sensors.add_argument("-w", "--watch", help="Poll and print GPU status", action="store_true")
    parser_sensors.set_defaults(func=sensors)

    parser_record = subparser.add_parser("record", description="Record GPU properties")
    parser_record.add_argument("-m", "--memory", help="Record amount of used GPU memory (default)", action="store_true")
    parser_record.add_argument("--autoscale", help="Enable auto-scaling of the y axis in the plot (default)",
                               action="store_true")
    parser_record.add_argument("--no-autoscale", help="Disable auto-scaling of the y axis in the plot",
                               dest='autoscale', action="store_false")
    parser_record.add_argument("-o", "--output", type=Path, default=None,
                               help="Path to save the generated plot which shows the recorded GPU property over time.")
    parser_record.add_argument("-p", "--plot",
                               help="Show the generated plot which shows the recorded GPU property over time.",
                               action="store_true")
    parser_record.set_defaults(func=record)
    parser_record.set_defaults(autoscale=True)
    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
