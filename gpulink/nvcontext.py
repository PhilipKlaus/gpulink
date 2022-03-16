import time

from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName


class NVContext:
    """
    Context for executing nvml queries.
    """

    def __init__(self):
        self._device_handles = []
        self._device_names = []
        self._valid_ctx = False

    def __enter__(self):
        nvmlInit()
        self._valid_ctx = True
        self._get_device_handles()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        nvmlShutdown()
        self._valid_ctx = False

    def _get_device_handles(self):
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            self._device_handles.append(handle)
            self._device_names.append(nvmlDeviceGetName(handle).decode("utf-8") )

    def execute_query(self, query, *args, **kwargs):
        if not self.valid_ctx:
            raise RuntimeError("Cannot execute query in an invalid NVContext")
        res = []
        for handle in self._device_handles:
            timestamp = time.time_ns()
            res.append((timestamp, query(handle, *args, **kwargs)))
        return res

    @property
    def valid_ctx(self):
        return self._valid_ctx

    @property
    def gpus(self):
        return len(self._device_handles)

    @property
    def gpu_names(self):
        return self._device_names
