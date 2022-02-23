import signal
import sys
import time
import numpy as np

from pynvml.smi import nvidia_smi
import matplotlib.pyplot as plt

measurements = []
resolution = 0.01

def get_mem_used(measurements):
    amount = len(nvsmi.DeviceQuery('gpu_name,gpu_bus_id,vbios_version')["gpu"])
    mem_consumed = [[] for gpu in range(amount)]

    for entry in measurements:
        measure = entry["gpu"]
        for gpu, data in enumerate(measure):
            mem_consumed[gpu].append(data["fb_memory_usage"]["total"] - data["fb_memory_usage"]["free"])

    return mem_consumed

def generate_report(mem_consumed):
    amount = len(mem_consumed[0])
    x = np.arange(0, amount * resolution, resolution)
    for i, data in enumerate(mem_consumed):
        plt.plot(x, data, label=f"GPU[{i}]")
    plt.title("Gpu memory usage [GB]")
    plt.legend(loc="upper left")
    plt.ylabel("Memory [GB]")
    plt.xlabel("Time [s]")
    plt.savefig("mem_consumption.png")

def signal_handler(signal, frame):
    mem_consumed = get_mem_used(measurements)
    mem_max = [max(m) for m in mem_consumed]
    generate_report(mem_consumed)
    for i, mem in enumerate(mem_max):
        print(f"GPU[{i}]: max memory consumption: {mem}[GB]")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    nvsmi = nvidia_smi.getInstance()
    while True:
        measurements.append(nvsmi.DeviceQuery('memory.free, memory.total'))
        time.sleep(resolution)

