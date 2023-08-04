import sys
import time

import gpulink as gpu


def get_device():
    if len(sys.argv) > 1 and sys.argv[1] == "mock":
        return gpu.DeviceMock
    return gpu.LocalNvmlGpu


if __name__ == "__main__":
    with gpu.DeviceCtx(device=get_device()) as ctx:
        # Record using the context manager
        recorder = gpu.Recorder.create_memory_recorder(
            ctx,
            name="Example Context Manager"
        )

        # This simulates GPU work
        with recorder:
            time.sleep(3)

        # Fetch and print recording
        recording = recorder.get_recording()
        print(recording)

        # Plot recorded data
        gpu.Plot(recording).plot()
