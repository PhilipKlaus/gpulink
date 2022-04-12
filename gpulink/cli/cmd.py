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
    parser_sensors.add_argument("-w", "--watch", help="Print regularly updated GPU stats", action="store_true")
    parser_sensors.set_defaults(func=sensors)

    parser_record = subparser.add_parser("record", description="Record GPU properties")
    parser_record.add_argument("-m", "--memory", help="Record used GPU memory (default)", action="store_true")
    parser_record.add_argument("-o", "--output", type=Path, default=None, help="Path to memory graph plot")
    parser_record.set_defaults(func=record)

    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
