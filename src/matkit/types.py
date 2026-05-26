"""Typed return value contracts for matkit functions.

These TypedDicts document the dict shapes returned by matkit's
core functions. They are purely for type-checking and IDE support
-- no runtime validation is performed.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class GRASPAResult(TypedDict):
    """Return type for ``matkit.graspa.get_output_data``."""

    success: bool
    uptake: Optional[float]
    error: Optional[float]
    unit: str
    qst: Optional[float]
    error_qst: Optional[float]
    qst_unit: str
    calc_time_in_s: Optional[float]


class GRASPASyclResult(TypedDict):
    """Return type for ``matkit.graspa_sycl.get_output_data``."""

    success: bool
    uptake: float
    error: float
    unit: str
    calc_time_in_s: Optional[float]


class RASPA2Result(TypedDict):
    """Return type for ``matkit.raspa2.get_output_data``."""

    success: bool
    uptake: float
    error: float
    unit: str
    calc_time_in_s: Optional[int]


class ZeoppResult(TypedDict):
    """Return type for ``matkit.zeopp.run_zeopp``."""

    success: bool
    results: dict
    error: Optional[str]


class UMABatchResult(TypedDict):
    """Record type for ``matkit.mlip.run_opt_uma_batch`` results."""

    structure: str
    model: str
    run_type: str
    task_name: str
    status: str  # "success", "failure", or "skipped"
    output_file: Optional[str]
    converged: Optional[bool]
    final_energy: Optional[float]
    n_steps: Optional[int]
    error_message: Optional[str]
