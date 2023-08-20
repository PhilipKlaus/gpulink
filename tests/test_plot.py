import numpy as np
import pytest

import gpulink as gpu
from gpulink.consts import SEC


@pytest.fixture
def device_ctx():
    return gpu.DeviceCtx(device=gpu.DeviceMock)


@pytest.fixture
def time_series():
    return [
        gpu.TimeSeries(
            np.array([i for i in range(10)]),
            np.array([i * 2 for i in range(10)])
        ),
        gpu.TimeSeries(
            np.array([i for i in range(10)]),
            np.array([i * 4 for i in range(10)])
        )
    ]


def test_generate_graph_with_autoscale(device_ctx, time_series):
    with device_ctx as ctx:
        recording = gpu.Recording(
            gpus=ctx.gpus,
            timeseries=time_series,
            rtype=gpu.RecType.REC_TYPE_TEMPERATURE,
            name="Test Recording",
            unit="°C"
        )

        plot = gpu.Plot(recording)
        _, ax = plot.generate_graph()

        gpu1 = ax.lines[0]
        gpu2 = ax.lines[1]
        time_expected = time_series[0].timestamps / SEC

        assert ax.get_title() == "Test Recording"
        assert ax.get_ylabel() == "Temperature [°C]"
        assert 0 > ax.get_ylim()[0] > -2
        assert 36 < ax.get_ylim()[1] < 38
        np.testing.assert_equal(gpu1.get_xdata(True), time_expected)
        np.testing.assert_equal(gpu2.get_xdata(True), time_expected)
        np.testing.assert_equal(gpu1.get_ydata(True), time_series[0].data)
        np.testing.assert_equal(gpu2.get_ydata(True), time_series[1].data)
