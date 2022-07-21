# gpulink

A library and command-line tool for monitoring NVIDIA GPU stats.  
**gpulink** uses [pynvml](https://github.com/gpuopenanalytics/pynvml) - a Python wrapper for
the [NVIDIA Management Library](https://developer.nvidia.com/nvidia-management-library-nvml) (NVML).

## Current status

**⚠️!!! This project is under heavy development - breaking changes between versions are possible!!!⚠️**

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
usage: gpulink [-h] {sensors,record} ...

GPU-Link: Monitor NVIDIA GPU status

positional arguments:
  {sensors,record}

optional arguments:
  -h, --help        show this help message and exit
```

### Examples

- View GPU sensor status: `gpulink sensors`
- Record the memory usage over time, generate a plot and save it as a png image: `gpulink record -o memory.png`

## Library usage

**gpulink** can be simply used from within Python. Just import `gpulink` and create a `DeviceCtx`. This context manages
device access and provides an API for fetching GPU properties
(see [API example](https://github.com/PhilipKlaus/gpu-link/blob/main/example/example_api.py)):

```
import gpulink as gpu

with gpu.DeviceCtx() as ctx:
   print(f"Available GPUs: {ctx.gpus.names}")
   memory_information = ctx.get_memory_info(gpus=ctx.gpus.ids)
```

### Recording data

**gpulink** provides a [Recorder](https://github.com/PhilipKlaus/gpu-link/blob/main/gpulink/recording/recorder.py) class
for recording GPU properties. For simple instantiation use one of the provided factory methods, e.g.:

```
    recorder = gpu.Recorder.create_memory_recorder(ctx, ctx.gpus.ids)
    recorder.start()
    ... # Do some GPU stuff
    recorder.stop(auto_join=True)
```

Once a recording is finished its data can be accessed:

```
recording = recording = recorder.get_recording()
```

### Plotting data

**gpulink** provides a [Plot](https://github.com/PhilipKlaus/gpu-link/blob/main/gpulink/plotting/plot.py) class for
visualizing recordings using [matplotlib](https://matplotlib.org/):

```
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

```
import gpulink as gpu

with gpu.DeviceCtx(device=DeviceMock) as ctx:
   ...
```