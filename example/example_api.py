import gpulink as gpu

with gpu.DeviceCtx() as ctx:
    ctx.get_memory_info(ctx.gpus.ids),
    ctx.get_fan_speed(gpus=ctx.gpus.ids),
    ctx.get_temperature(gpu.TemperatureSensorType.GPU, ctx.gpus.ids),
    ctx.get_temperature_threshold(gpu.TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX, ctx.gpus.ids),
    ctx.get_clock(gpu.ClockType.CLOCK_MEM, gpus=ctx.gpus.ids),
    ctx.get_power_usage(ctx.gpus.ids)
