# gpulink

A library for monitoring and displaying NVIDIA GPU stats.  
**gpulink** uses [pynvml](https://github.com/gpuopenanalytics/pynvml) - a Python wrapper for
the [NVIDIA Management Library](https://developer.nvidia.com/nvidia-management-library-nvml) (NVML).

## Current status

**⚠️!!! This project is under heavy development !!!⚠️**

## Installation

### Installation using PIP

To install **gpulink** using the Python Package Manager (PIP) run:  
```pip install gpulink```

**gpulink** can also be used from source. For this, perform the following steps to create a Python environment and to
install the requirements:

1. Create an environment: `python -m venv env`
2. Activate the environment: `.\env\Scripts\Activate`
3. Install requirements: `pip install -r requirements.txt`

## Usage

**gpulink** can either be used from the command line or imported as a library.

## Library usage

To integrate **gpulink** to a Python script, import `gpulink` and create an `NVContext`. This context manages the
creation and destruction of the nvml session and provides several query and utility functions (
see [API example](https://github.com/PhilipKlaus/gpu-link/blob/main/example/example_api.py)):

```
import gpulink as gpu

with gpu.NVContext() as ctx:
   print(f"Available GPUs: {ctx.gpu_names}")
   memory_information = ctx.get_memory_info(ctx.gpus)
   ...
```

**gpulink** provides a [Recorder](https://github.com/PhilipKlaus/gpu-link/blob/main/gpulink/recorder.py) class for recording
several GPU properties. An instance of this class must be created using one of the factory methods, e.g.:

```
    recorder = gpu.Recorder.create_memory_recorder(ctx, ctx.gpus)
    recorder.start()
    ... # Do some GPU stuff
    recorder.stop(auto_join=True)
```

Once a recording is finished, the data can be accessed:

```
recording = recording = recorder.get_recording()
```

**gpulink** provides a [Plot](https://github.com/PhilipKlaus/gpu-link/blob/main/gpulink/plot.py) class for visualizing recordings
using [matplotlib](https://matplotlib.org/):

```
    from pathlib import Path
    
    plot = gpu.Plot(recording)
    plot.plot(scale_y_axis=True)
    plot.save(Path("memory.png"), scale_y_axis=True)
    
    figure, axis = plot.generate_graph()  # The generated Figure and Axis can also be accessed directly.
```

## Command-line usage

During installation, **gpulink** also registers a command-line script accessible through the `gpulink` command.

```
usage: gpulink [-h] {sensors,record} ...

GPU-Link: Monitor NVIDIA GPU status

positional arguments:
  {sensors,record}

optional arguments:
  -h, --help        show this help message and exit
```

### Examples

- View GPU sensor status: `gpulink sensors`
```
╒════════╤═════════════════════╤═════════════╤═════════════════╤═══════════════╕
│ GPU    │ Memory [MB]         │   Temp [°C] │   Fan speed [%] │ Clock [MHz]   │
╞════════╪═════════════════════╪═════════════╪═════════════════╪═══════════════╡
│ GPU[0] │ 1588 / 25769 (6.2%) │          34 │              41 │ Graph.: 173   │
│        │                     │             │                 │ Memory: 403   │
│        │                     │             │                 │ SM: 173       │
│        │                     │             │                 │ Video: 539    │
╘════════╧═════════════════════╧═════════════╧═════════════════╧═══════════════╛
```
- Record gpu memory information and save a plot as PNG: `gpulink record -o memory.png`
```
╒═════════════════════╤═════════════════╕
│ Record duration [s] │ Frame rate [Hz] │
├─────────────────────┼─────────────────┤
│ 14.0294178          │ 235             │
╘═════════════════════╧═════════════════╛
╒═════╤══════════════════╤══════════════════════╕
│ GPU │ Name             │ Memory used [MB]     │
├─────┼──────────────────┼──────────────────────┤
│ 0   │ NVIDIA TITAN RTX │ minimum: 1584.754688 │
│     │                  │ maximum: 2204.585984 │
╘═════╧══════════════════╧══════════════════════╛
```
![Memory consumption over time](https://github.com/PhilipKlaus/gpu-link/blob/main/docs/mem_consumption.png)
