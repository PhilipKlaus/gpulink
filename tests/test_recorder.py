import pytest

from gpulink import DeviceCtx, Recorder
from gpulink.types import RecType, TimeSeries, PlotInfo
from tests.misc import DeviceMock, GB


@pytest.fixture
def device_ctx():
    return DeviceCtx(device_type=DeviceMock)


def test_get_record(device_ctx):
    with device_ctx as ctx:
        rec = Recorder(
            cmd=lambda c: c.get_memory_info(),
            res_filter=lambda res: res.used,
            ctx=ctx,
            rec_type=RecType.MEMORY_USED,
            gpus=ctx.gpus.ids
        )
        assert rec.get_record() == ([0, 0], [GB // 2, GB // 2])


def test_fetch_and_return_data(device_ctx):
    with device_ctx as ctx:
        rec = Recorder(
            cmd=lambda c: c.get_memory_info(),
            res_filter=lambda res: res.used,
            ctx=ctx,
            rec_type=RecType.MEMORY_USED,
            gpus=ctx.gpus.ids
        )
        for i in range(3):
            rec.fetch_and_store()

        data = rec.get_recording()
        assert data.rec_type == RecType.MEMORY_USED
        assert data.gpus == ctx.gpus
        assert data.timeseries == [
            TimeSeries([0, 0, 0], [GB // 2, GB // 2, GB // 2]),
            TimeSeries([0, 0, 0], [GB // 2, GB // 2, GB // 2])
        ]
        assert data.plot_info == PlotInfo(max_values=[GB, GB])
