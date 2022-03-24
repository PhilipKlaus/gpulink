# gpulink

A library for monitoring and displaying NVIDIA GPU stats. For this purpose
**gpulink** provides classes for data recording and plotting, targeting different GPU stats (e.g. memory consumption).
**gpulink** makes use of [pynvml](https://github.com/gpuopenanalytics/pynvml) - a Python wrapper for
the [NVIDIA Management Library](https://developer.nvidia.com/nvidia-management-library-nvml) (NVML).

| Feature        | Status | Notes |
|----------------|--------|-------|
| MemoryRecorder | ✅      |       |
| MemoryPlotter  | ✅      |       |

During installation, **gpulink** also registers a command-line script accessible through the `gpulink` command.

## Library usage

To integrate **gpulink** to a Python script, import `gpulink` and create an `NVContext`. This context manages the creation and
destruction of the nvml session and provides several utility functions:

```
import gpulink as gpu

with gpu.NVContext() as ctx:
```

Every subsequent **gpulink** call (e.g. instantiation of a data recorder) must be made within this context. The context
provides also additional information, i.e. about the numbers and names of compatible GPU devices:

```
    print(f"Available GPUs: {ctx.gpu_names}")
```

**gpulink** provides data recorders (e.g. `MemoryRecorder`) for capturing and storing GPU stats and plotters (e.g.
`MemoryPlotter`) for rendering these:

```
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
    graph.draw_graph()
```

## Command-line usage

```
usage: gpu-link.py [-h] [-o OUTPUT]

GPU-Link memory tracing

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to memory graph plot.
```

### Example
1. Run recording  
(a) Installed from source: `python -m gpulink -o memory_consumption.png`  
(b) Installed using pip: `gpulink -o memory_consumption.png`
2. Stop recording *Ctrl+C*
3. Plot is saved to `memory_consumption.png`.

### Output
```
NVIDIA TITAN RTX[0] -> memory used: min=1729.236992[MB] / max=2315.79648[MB]
```
