import gpulink as gpu

with gpu.DeviceCtx(device=gpu.DeviceMock) as ctx:
    ctx.get_memory_info(gpus=ctx.gpus.ids),
    ctx.get_fan_speed(gpus=ctx.gpus.ids),
    ctx.get_temperature(gpu.TemperatureSensorType.GPU, gpus=ctx.gpus.ids),
    ctx.get_temperature_threshold(gpu.TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX, gpus=ctx.gpus.ids),
    ctx.get_clock(gpu.ClockType.CLOCK_MEM, gpus=ctx.gpus.ids),
    ctx.get_power_usage(gpus=ctx.gpus.ids)
