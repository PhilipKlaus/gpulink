import time

import gpulink as gpu

with gpu.DeviceCtx() as ctx:
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
