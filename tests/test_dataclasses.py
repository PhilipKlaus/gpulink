from gpulink.gpu_types import PlotOptions


def test_patch_plot_options():
    old = PlotOptions()
    old.patch(PlotOptions(
        plot_name="Foo Bar",
        y_axis_range=(-100, 100),
        y_axis_unit="°C",
        y_axis_label="Temperature",
        y_axis_divider=4
    ))
    assert old.plot_name == "Foo Bar"
    assert old.y_axis_range == (-100, 100)
    assert old.y_axis_unit == "°C"
    assert old.y_axis_label == "Temperature"
    assert old.y_axis_divider == 4
