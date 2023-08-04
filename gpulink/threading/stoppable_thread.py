from threading import Thread, Event


class StoppableThread(Thread):
    def __init__(self):
        super().__init__()
        self._stop_event = Event()

    def stop(self, auto_join=True) -> None:
        self._stop_event.set()
        if auto_join:
            self.join()

    @property
    def should_stop(self):
        return self._stop_event.is_set()
