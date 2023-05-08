import itertools

from colorama import Cursor


def set_cursor(x: int, y: int):
    return Cursor.POS(x, y)


def get_spinner() -> itertools.cycle:
    return itertools.cycle(["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"])
