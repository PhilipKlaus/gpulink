import sys

import gpulink as gpu


def get_device():
    if len(sys.argv) > 1 and sys.argv[1] == "mock":
        return gpu.DeviceMock
    return gpu.LocalNvmlGpu


if __name__ == "__main__":
    with gpu.DeviceCtx(device=get_device()) as ctx:
        ctx.get_memory_info(gpus=ctx.gpus.ids),
        ctx.get_fan_speed(gpus=ctx.gpus.ids),
        ctx.get_temperature(gpu.TemperatureSensorType.GPU, gpus=ctx.gpus.ids),
        ctx.get_temperature_threshold(gpu.TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX, gpus=ctx.gpus.ids),
        ctx.get_clock(gpu.ClockType.CLOCK_MEM, gpus=ctx.gpus.ids),
        ctx.get_power_usage(gpus=ctx.gpus.ids)
