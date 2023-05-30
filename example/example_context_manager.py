import time

import gpulink as gpu

with gpu.DeviceCtx() as ctx:
    # Record using the context manager
    recorder = gpu.Recorder.create_memory_recorder(
        ctx,
        ctx.gpus.ids,
        plot_options=gpu.PlotOptions(
            plot_name="Example Basic",
        )
    )

    # This simulates GPU work
    with recorder:
        time.sleep(3)

    # Fetch and print recording
    recording = recorder.get_recording()
    print(recording)

    # Plot recorded data
    gpu.Plot(recording).plot()
