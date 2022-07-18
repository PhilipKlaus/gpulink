import numpy as np
import pytest

from gpulink import DeviceCtx, Plot
from gpulink.consts import SEC
from gpulink.record.timeseries import TimeSeries
from gpulink.record.recording import Recording
from gpulink.plotting.plot_options import PlotOptions
from tests.misc import DeviceMock


@pytest.fixture
def device_ctx():
    return DeviceCtx(device=DeviceMock)


@pytest.fixture
def time_series():
    return [
        TimeSeries(
            [i for i in range(10)],
            [i * 2 for i in range(10)]
        ),
        TimeSeries(
            [i for i in range(10)],
            [i * 4 for i in range(10)]
        )
    ]


def test_generate_graph_without_plot_options(device_ctx, time_series):
    with device_ctx as ctx:
        recording = Recording(gpus=ctx.gpus, timeseries=time_series, plot_options=None)

        plot = Plot(recording)
        _, ax = plot.generate_graph()

        gpu1 = ax.lines[0]
        gpu2 = ax.lines[1]
        time_expected = time_series[0].timestamps / SEC

        assert ax.get_title() == "GPULink Recording"
        np.testing.assert_equal(gpu1.get_xdata(True), time_expected)
        np.testing.assert_equal(gpu2.get_xdata(True), time_expected)
        np.testing.assert_equal(gpu1.get_ydata(True), time_series[0].data)
        np.testing.assert_equal(gpu2.get_ydata(True), time_series[1].data)


def test_generate_graph_with_custom_plot_options(device_ctx, time_series):
    with device_ctx as ctx:
        recording = Recording(
            gpus=ctx.gpus,
            timeseries=time_series,
            plot_options=PlotOptions(
                plot_name="GPULink Recording: Foo Bar",
                y_axis_unit="°C",
                y_axis_label="Temperature",
                y_axis_divider=2,
                y_axis_range=(-100, 100)
            )
        )

        plot = Plot(recording)
        _, ax = plot.generate_graph()

        gpu1 = ax.lines[0]
        gpu2 = ax.lines[1]

        assert ax.get_title() == "GPULink Recording: Foo Bar"
        assert ax.get_ylabel() == "Temperature [°C]"
        assert ax.get_ylim()[0] == -100 / 2
        assert ax.get_ylim()[1] == 100 / 2
        np.testing.assert_equal(gpu1.get_ydata(True), time_series[0].data / 2)
        np.testing.assert_equal(gpu2.get_ydata(True), time_series[1].data / 2)
