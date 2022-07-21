import itertools
import os
import sys

from colorama import Cursor


def cls():
    os.system("cls")


def set_cursor(x: int, y: int):
    return Cursor.POS(x, y)


def get_spinner() -> itertools.cycle:
    return itertools.cycle(['-', '\\', '|', '/'])


def print_and_clear(msg: str) -> None:
    erase_list = ['\b' for _ in msg]
    erase = "".join(erase_list)

    sys.stdout.write(f"{msg}")
    sys.stdout.flush()
    sys.stdout.write(erase)
