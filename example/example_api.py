import gpulink as gpu

with gpu.NVContext() as ctx:
    ctx.get_memory_info(ctx.gpus),
    ctx.get_fan_speed(ctx.gpus),
    ctx.get_temperature(ctx.gpus, gpu.TemperatureSensorType.GPU),
    ctx.get_temperature_threshold(ctx.gpus, gpu.TemperatureThreshold.TEMPERATURE_THRESHOLD_GPU_MAX),
    ctx.get_clock(ctx.gpus, gpu.ClockType.CLOCK_MEM),
    ctx.get_power_usage(ctx.gpus)