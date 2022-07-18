import time

import gpulink as gpu

with gpu.DeviceCtx() as ctx:
    print(f"Available GPUs (names): {ctx.gpus.names}")
    print(f"Available GPUs (ids): {ctx.gpus.ids}")
    print(f"Single Memory Info query: {ctx.get_memory_info(ctx.gpus.ids)}")

    # Create recorder and record the amount of used GPU memory for 3 seconds
    recorder = gpu.Recorder.create_memory_recorder(ctx, ctx.gpus.ids)
    recorder.start()
    time.sleep(3)
    recorder.stop(auto_join=True)

    # Fetch and print recording
    recording = recorder.get_recording()
    print(recording)

    # Plot recorded data
    gpu.Plot(recording).plot()
