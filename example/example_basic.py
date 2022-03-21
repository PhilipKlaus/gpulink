import gpulink as gpu

with gpu.NVContext() as ctx:
    print(f"Available GPUs: {ctx.gpu_names}")

    recorder = gpu.MemoryRecorder(ctx)
    mem_info = recorder.record(store=False)

    for _ in range(5):  # Record memory information over time
        recorder.record()

    records = recorder.get_records()
    for rec in records:
        print(f"Sampling rate: {rec.sampling_rate}")
        print(f"Recording duration: {rec.duration}")
        print(f"Number of records: {rec.len}")
        print(rec)

    graph = gpu.MemoryPlotter(records)
    graph.plot()
