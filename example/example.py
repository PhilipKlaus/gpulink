import gpulink as gpu

with gpu.NVContext() as ctx:
    print(f"Available GPUs: {ctx.gpu_names}")
    recorder = gpu.MemoryRecorder(ctx)
    init_mem = recorder.memory_info  # Get actual memory information

    for _ in range(5):  # Record memory information over time
        recorder.record()

    print(f"Memory info at start: {init_mem}")
    print(f"Recorded {recorder.num_records} data frames")
    print(recorder.records)
