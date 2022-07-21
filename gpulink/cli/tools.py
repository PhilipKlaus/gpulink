import signal
import time
from typing import Optional, Callable

from gpulink.cli.console import get_spinner, print_and_clear, cls
from gpulink.threading.stoppable_thread import StoppableThread


class _CliInterrupt:

    def __init__(self, interrupt_signal: int, task: Callable):
        signal.signal(interrupt_signal, self._interrupt)
        self._interrupt_task = task

    def _interrupt(self, _, __) -> None:
        if self._interrupt_task is not None:
            self._interrupt_task()


def start_in_background(thread: StoppableThread, waiting_msg: Optional[str] = None) -> None:
    _ = _CliInterrupt(signal.SIGINT, lambda: thread.stop(auto_join=True))

    spinner = None
    if waiting_msg:
        spinner = get_spinner()
        cls()

    thread.start()
    while thread.is_alive():
        if waiting_msg:
            print_and_clear(f"{waiting_msg} {next(spinner)}")
        time.sleep(0.1)

    if waiting_msg:
        cls()
