from enum import Enum

SEC = 1e9

MB = 1e6  # Megabyte
GB = 1e9  # Gigabyte
WATTS = 1e3  # Watts


class RecType(Enum):
    REC_TYPE_CLOCK_GRAPHICS = "Graphics Clock"
    REC_TYPE_CLOCK_SM = "Shared Memory Clock"
    REC_TYPE_CLOCK_MEM = "Memory Clock"
    REC_TYPE_CLOCK_VIDEO = "Video Clock"
    REC_TYPE_POWER_USAGE = "Power Usage"
    REC_TYPE_FAN_SPEED = "Fan Speed"
    REC_TYPE_TEMPERATURE = "Temperature"
    REC_TYPE_MEMORY = "Memory"
