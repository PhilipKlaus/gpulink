from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class QueryResult:
    """
    Represents the base class for a single GPU query result.
    """
    timestamp: int
    gpu_idx: int
    gpu_name: str


@dataclass
class SimpleResult(QueryResult):
    """
    A GPUQueryResult containing only a single value.
    """
    value: Union[int, float, str]


@dataclass
class MemInfo(QueryResult):
    """
    Stores the result of a nvmlDeviceGetMemoryInfo query.
    """
    total: int
    used: int
    free: int
