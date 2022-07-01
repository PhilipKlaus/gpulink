import gpulink as gpu

with gpu.DeviceCtx() as ctx:
    ctx.get_memory_info(ctx.gpus.ids),
    ctx.get_fan_speed(ctx.gpus.ids),
    ctx.get_temperature(ctx.gpus.ids, gpu.TemperatureSensorType.GPU),
    ctx.get_temperature_threshold(ctx.gpus.ids, gpu.TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX),
    ctx.get_clock(ctx.gpus.ids, gpu.ClockType.CLOCK_MEM),
    ctx.get_power_usage(ctx.gpus.ids)
