from __future__ import annotations

from typing import Callable

import numpy as np


class TimeSeries:
    def __init__(self, timestamps: np.ndarray, data: np.ndarray):
        self._timestamps = timestamps
        self._data = data

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, TimeSeries):
            return np.array_equal(self.timestamps, other.timestamps) and np.array_equal(self.data, other.data)
        return False

    def apply_to_data(self, fn: Callable[[np.ndarray], np.ndarray]):
        self._data = fn(self._data)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array(self._timestamps)

    @property
    def data(self) -> np.ndarray:
        return np.array(self._data)
