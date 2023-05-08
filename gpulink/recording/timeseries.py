from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Union

import numpy as np


@dataclass
class TimeSeries:
    def __init__(self, timestamps: Optional[List[int]] = None, data: Optional[List[Union[int, float]]] = None):
        if data is None:
            data = []
        if timestamps is None:
            timestamps = []
        self._timestamps = timestamps
        self._data = data

    def add_record(self, timestamp, data):
        self._timestamps.append(timestamp)
        self._data.append(data)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array(self._timestamps)

    @property
    def data(self) -> np.ndarray:
        return np.array(self._data)
