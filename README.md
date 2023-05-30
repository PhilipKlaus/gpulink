# gpulink

[![Downloads](https://static.pepy.tech/badge/gpulink/week)](https://pepy.tech/project/gpulink)
![PythonTest](https://github.com/PhilipKlaus/gpulink/actions/workflows/python-test.yml/badge.svg)

A library and command-line tool for monitoring NVIDIA GPU stats.  
**gpulink** uses [pynvml](https://github.com/gpuopenanalytics/pynvml) - a Python wrapper for
the [NVIDIA Management Library](https://developer.nvidia.com/nvidia-management-library-nvml) (NVML).

## Current status

⚠ gpulink is in a very early state - breaking changes between versions are possible!

## Requirements

**gpulink** requires the NVIDIA Management Library to be installed which is shipped together
with [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface).

## Installation

### Installation using PIP

To install **gpulink** using the Python Package Manager (PIP) run:  
```pip install gpulink```

### Using from source

**gpulink** can also be used from source. For this, perform the following steps to create a Python environment and to
install the requirements:

1. Create an environment: `python -m venv env`
2. Activate the environment: `.\env\Scripts\Activate`
3. Install requirements: `pip install -r requirements.txt`

## Command-line usage

**gpulink** can either be imported as a library or can be used from the command line:

```
Usage: GPU-Link: Monitor NVIDIA GPUs [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  record   Record GPU properties.
  sensors  Fetch and print the GPU sensor status.
```

### Examples

- View GPU sensor status: `gpulink sensors`

```
╒═══════╤══════════════════╤═════════════════════╤═════════════╤═════════════════╤═══════════════╤═══════════════════╕
│   GPU │ Name             │ Memory [MB]         │   Temp [°C] │   Fan speed [%] │ Clock [MHz]   │   Power Usage [W] │
╞═══════╪══════════════════╪═════════════════════╪═════════════╪═════════════════╪═══════════════╪═══════════════════╡
│     0 │ NVIDIA TITAN RTX │ 1809 / 25769 (7.0%) │          34 │              41 │ Graph.: 173   │            26.583 │
│       │                  │                     │             │                 │ Memory: 403   │                   │
│       │                  │                     │             │                 │ SM: 173       │                   │
│       │                  │                     │             │                 │ Video: 540    │                   │
╘═══════╧══════════════════╧═════════════════════╧═════════════╧═════════════════╧═══════════════╧═══════════════════╛
```

- Watch GPU sensor status: `gpulink sensors -w`

![Watch sensor status](https://github.com/PhilipKlaus/gpu-link/blob/main/docs/gpulink_sensors_watch.gif)

- Record the memory usage over time, generate a plot and save it as a png image: `gpulink record -o memory.png memory`

```
╒═════╤══════════════════╤══════════════════════╕
│ GPU │ Name             │ Memory used [MB]     │
├─────┼──────────────────┼──────────────────────┤
│ 0   │ NVIDIA TITAN RTX │ minimum: 1584.754688 │
│     │                  │ maximum: 2204.585984 │
╘═════╧══════════════════╧══════════════════════╛
Duration:       2.500       [s]"
Sampling rate:  300.000     [Hz]"
```

![Memory consumption over time](https://github.com/PhilipKlaus/gpu-link/blob/main/docs/mem_consumption.png)

## Library usage

**gpulink** can be simply used from within Python. Just import `gpulink` and create a `DeviceCtx`. This context manages
device access and provides an API for fetching GPU properties
(see [API example](https://github.com/PhilipKlaus/gpu-link/blob/main/example/example_api.py)):

``` python
import gpulink as gpu

with gpu.DeviceCtx() as ctx:
   print(f"Available GPUs: {ctx.gpus.names}")
   memory_information = ctx.get_memory_info(gpus=ctx.gpus.ids)
```

### Recording data

**gpulink** provides a [Recorder](https://github.com/PhilipKlaus/gpu-link/blob/main/gpulink/recording/recorder.py) class
for recording GPU properties. For simple instantiation use one of the provided factory methods, e.g.:

``` python
recorder = gpu.Recorder.create_memory_recorder(ctx, ctx.gpus.ids)
```

Afterwards a recording can be performed:

#### Option 1: Using `start` and `stop` method (see
[Basic example](https://github.com/PhilipKlaus/gpu-link/blob/main/example/example_basic.py))

``` python
    recorder.start()
    ... # Do some GPU stuff
    recorder.stop(auto_join=True)
```

#### Option 2: Using a context manager (see
[Context-Manager example](https://github.com/PhilipKlaus/gpu-link/blob/main/example/example_context_manager.py))

``` python
    with recorder:
    ... # Do some GPU stuff
```

#### Option 3: Using a decorator (see
[Decorator example](https://github.com/PhilipKlaus/gpu-link/blob/main/example/example_decorator.py))

``` python
    @record(factory=gpu.Recorder.create_memory_recorder)
    def my_gpu_function():
    ... # Do dome GPU stuff
    
    my_gpu_function()
```

Once a recording is finished its data can be accessed:

``` python
recording = recording = recorder.get_recording()
```

### Plotting data

**gpulink** provides a [Plot](https://github.com/PhilipKlaus/gpu-link/blob/main/gpulink/plotting/plot.py) class for
visualizing recordings using [matplotlib](https://matplotlib.org/):

``` python
    from pathlib import Path
    
    # Generate the plot
    plot = gpu.Plot(recording)
    
    # Display the plot
    plot.plot()
    
    # Save the plot as an image
    plot.save(Path("memory.png"))
    
    # The generated Figure and Axis can also be accessed directly
    figure, axis = plot.generate_graph()
```

The plot can be parametrized using
the [PlotOptions](https://github.com/PhilipKlaus/gpu-link/blob/main/gpulink/plotting/plot_options.py) dataclass. An
example using custom plot options is given
in [Basic example](https://github.com/PhilipKlaus/gpu-link/blob/main/example/example_basic.py)

## Unit testing

When using **gpulink** inside unit tests, create or use an already existing device mock,
e.g. [DeviceMock](https://github.com/PhilipKlaus/gpu-link/blob/main/gpulink/tests/device_mock.py). Then during creating
a `DeviceCtx` provide the mock as follows:

``` python
import gpulink as gpu

with gpu.DeviceCtx(device=DeviceMock) as ctx:
   ...
```

## Troubleshooting

- If you get the error message below, please ensure that the NVIDIA Management Library is installed on you system by
  typing `nvidia-smi --version` into a terminal:  
  ```pynvml.nvml.NVMLError_LibraryNotFound: NVML Shared Library Not Found```.

## Planned features

- Live-plotting of GPU stats

## Changelog

- **0.4.0**
    - Recording arbitrary GPU stats (clock, fan-speed, memory, power-usage, temp)
    - Display GPU name and power usage within `sensors` command
    - Replaced `arparse` library by [click](https://click.palletsprojects.com/en/8.1.x/)
    - Aborting a `watch` or `recording` command can be done by pressing any instead of `ctrl+c`
- **0.4.1**
    - Fix error when calling `nvmlDeviceGetName` in `pynvml` version *11.5.0*
- **0.5.0**
    - Add context-manager-based recording
    - Add decorator-based recording
