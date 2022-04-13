import itertools
import signal
import sys
import time
from typing import Optional, Callable

from gpulink.stoppable_thread import StoppableThread


def get_spinner() -> itertools.cycle:
    return itertools.cycle(['-', '\\', '|', '/'])


def _print_and_clear(msg: str) -> None:
    erase_list = ['\b' for _ in msg]
    erase = "".join(erase_list)

    sys.stdout.write(f"{msg}")
    sys.stdout.flush()
    sys.stdout.write(erase)


class _CliInterrupt:

    def __init__(self, interrupt_signal: int, task: Callable):
        signal.signal(interrupt_signal, self._interrupt)
        self._interrupt_task = task

    def _interrupt(self, _, __) -> None:
        if self._interrupt_task is not None:
            self._interrupt_task()


def busy_wait_for_interrupt(thread: StoppableThread, waiting_msg: Optional[str] = None) -> None:
    _ = _CliInterrupt(signal.SIGINT, lambda: thread.stop(auto_join=True))

    spinner = None
    if waiting_msg:
        spinner = get_spinner()

    thread.start()
    while thread.is_alive():
        if waiting_msg:
            _print_and_clear(f"{waiting_msg} {next(spinner)}")
        time.sleep(0.1)
