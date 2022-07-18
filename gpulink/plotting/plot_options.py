from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Union, Optional, Tuple

DATA = Union[int, float]


@dataclass
class PlotOptions:
    plot_name: Optional[str] = None
    y_axis_range: Optional[Tuple[DATA, DATA]] = None
    y_axis_label: Optional[str] = None
    y_axis_unit: Optional[str] = None
    y_axis_divider: Optional[DATA] = None

    def patch(self, other: PlotOptions):
        for field in fields(PlotOptions):
            value = other.__getattribute__(field.name)
            if value:
                self.__setattr__(field.name, value)