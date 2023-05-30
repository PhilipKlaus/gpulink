import time

import gpulink as gpu


# Use a decorator for recording with standard settings:
# ctx_class=LocalNvmlGpu, gpus=None (means all GPUs)
@gpu.record(factory=gpu.Recorder.create_memory_recorder)
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
