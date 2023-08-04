import sys
import time

import gpulink as gpu


def get_device():
    if len(sys.argv) > 1 and sys.argv[1] == "mock":
        return gpu.DeviceMock
    return gpu.LocalNvmlGpu


if __name__ == "__main__":
    # Use a decorator for recording with standard settings:
    @gpu.record(rtype=gpu.RecType.REC_TYPE_MEMORY, ctx_class=gpu.DeviceMock)
    def my_gpu_function(a: int, b: int):
        time.sleep(3)
        return a + b


    result = my_gpu_function(10, 20)

    # Fetch and print return value from function
    value = result.value
    print(f"Function return value: {value}")

    # Fetch and print recording
    recording = result.recording
    print(recording)

    # Plot recorded data
    gpu.Plot(recording).plot()
