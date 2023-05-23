import time

import pytest

from gpulink import DeviceCtx, Recorder
from gpulink.consts import MB
from gpulink.plotting.plot_options import PlotOptions
from gpulink.recording.timeseries import TimeSeries
from tests.device_mock import DeviceMock, TEST_GB


@pytest.fixture
def device_ctx():
    return DeviceCtx(device=DeviceMock)


def test_get_record(device_ctx):
    with device_ctx as ctx:
        rec = Recorder.create_memory_recorder(ctx, ctx.gpus.ids)
        assert rec._get_record() == ([0, 0], [TEST_GB // 2, TEST_GB // 2])


def test_fetch_and_return_data(device_ctx):
    with device_ctx as ctx:
        rec = Recorder.create_memory_recorder(ctx, ctx.gpus.ids)

        for i in range(3):
            rec._fetch_and_store()

        data = rec.get_recording()
        assert data.gpus == ctx.gpus
        assert data.timeseries == [
            TimeSeries([0, 0, 0], [TEST_GB // 2, TEST_GB // 2, TEST_GB // 2]),
            TimeSeries([0, 0, 0], [TEST_GB // 2, TEST_GB // 2, TEST_GB // 2])
        ]

        assert data.plot_options == PlotOptions(
            plot_name="Memory usage",
            y_axis_unit="MB",
            y_axis_range=(0, TEST_GB),
            y_axis_divider=MB,
            y_axis_label="Memory usage",
            auto_scale=True
        )


def test_recorder_thread_with_custom_plot_options(device_ctx):
    with device_ctx as ctx:
        options = PlotOptions(
            plot_name="Custom Memory Usage",
            y_axis_divider=42,
            y_axis_unit="°C",
            y_axis_label="Foo Bar",
            y_axis_range=(0, 42),
            auto_scale=True
        )
        rec = Recorder.create_memory_recorder(
            ctx,
            ctx.gpus.ids,
            plot_options=options
        )
        rec.start()
        time.sleep(1)
        rec.stop(auto_join=True)

        assert rec.get_recording().plot_options == options


def test_record_using_context_manager(device_ctx):
    with device_ctx as ctx:
        rec = Recorder.create_memory_recorder(ctx, ctx.gpus.ids)

        # Ensure that the recording is empty at the beginning
        recording = rec.get_recording()
        assert recording.timeseries == [TimeSeries(), TimeSeries()]

        with rec:
            time.sleep(1)

        # Ensure that the recording is not empty after recording for 1 second
        recording = rec.get_recording()
        assert recording.timeseries[0].data.shape[0] > 0
        assert recording.timeseries[1].data.shape[0] > 0
