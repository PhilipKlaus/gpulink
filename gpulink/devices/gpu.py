from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Gpu:
    """
    Represents the base class for a single GPU query result.
    """
    id: int
    name: str


class GpuSet:
    """
    A set of GPUs.
    """
    def __init__(self, gpus: List[Gpu]):
        self._gpus = gpus

    def __getitem__(self, key) -> Gpu:
        return self._gpus[key]

    def __len__(self) -> int:
        return len(self._gpus)

    def __eq__(self, other):
        return self.ids == other.ids and self.names == other.names

    @property
    def ids(self) -> List[int]:
        return [gpu.id for gpu in self._gpus]

    @property
    def names(self) -> List[str]:
        return [gpu.name for gpu in self._gpus]