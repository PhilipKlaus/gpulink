import time

import gpulink as gpu
from gpulink.plotting.plot_options import PlotOptions

with gpu.DeviceCtx() as ctx:
    print(f"Available GPUs (names): {ctx.gpus.names}")
    print(f"Available GPUs (ids): {ctx.gpus.ids}")
    print(f"Single Memory Info query: {ctx.get_memory_info(ctx.gpus.ids)}")


    def create_recorder():
        # Create GPU memory recorder
        return gpu.Recorder.create_memory_recorder(
            ctx,
            ctx.gpus.ids,
            plot_options=PlotOptions(
                plot_name="Example Basic",
            )
        )


    # OPTION 1: Record using 'start' and 'stop' methods
    recorder = create_recorder()
    recorder.start()
    time.sleep(3)
    recorder.stop(auto_join=True)

    # OPTION 2: Record using the context manager
    recorder = create_recorder()
    with recorder:
        time.sleep(3)

    # Fetch and print recording
    recording = recorder.get_recording()
    print(recording)

    # Plot recorded data
    gpu.Plot(recording).plot()
