import time

import gpulink as gpu
from gpulink.recorder import Recorder, RecType

with gpu.NVContext() as ctx:

    print(f"Available GPUs: {ctx.gpus}")
    print(f"Single Memory Info query: {ctx.get_memory_info(ctx.gpus)}")

    recorder = Recorder.create_memory_recorder(ctx, ctx.gpus)
    recorder.start()
    time.sleep(3)  # Record for 3 seconds
    recorder.stop()
    recorder.join()

    recording = recorder.get_recording()
    print(recording)
