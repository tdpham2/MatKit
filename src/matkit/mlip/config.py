"""Configuration objects for runtime-selectable MLIP calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Integral, Real
from typing import Any, Literal, TypeAlias


_DTYPES = {"float32", "float64"}
_OPTIMIZERS = {"bfgs", "lbfgs", "gpmin", "fire", "mdmin"}


def _positive_number(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")


def _positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be at least 1 and an integer")


def _validate_explicit_options(
    backend: str, driver: str, calculator_type: str, provided: set[str]
) -> None:
    """Reject explicit frontend options the selected calculation cannot use."""
    backend_options = {
        "ase-mace": {"calculator_type", "dispersion"},
        "rootstock": {
            "cluster",
            "root_path",
            "cache_root",
            "setup_kwarg",
            "timeout",
            "weights",
        },
        "nvalchemi-mace": {
            "dt",
            "compile_model",
            "enable_cueq",
            "batch_size",
            "max_atoms",
        },
    }
    for owner, names in backend_options.items():
        for name in sorted(provided & names):
            if backend != owner:
                flag = "root" if name == "root_path" else name.replace("_", "-")
                raise ValueError(f"--{flag} requires --backend {owner}")
    if "dtype" in provided:
        if backend == "rootstock":
            raise ValueError(
                "--dtype is not supported by Rootstock; use --setup-kwarg "
                "with precision settings supported by the deployed model"
            )
        if backend == "ase-mace" and calculator_type == "mace_anicc":
            raise ValueError("--dtype is controlled by the mace_anicc factory")
    if "dispersion" in provided and calculator_type != "mace_mp":
        raise ValueError("--dispersion requires --calculator-type mace_mp")
    if driver == "energy":
        opt_only = provided & {"optimizer", "fmax", "steps", "dt"}
        if opt_only:
            flag = sorted(opt_only)[0].replace("_", "-")
            raise ValueError(f"--{flag} requires --driver opt")


@dataclass(frozen=True)
class ASEMACEConfig:
    """Run a MACE calculator directly through ASE."""

    checkpoint: str = "medium"
    device: str = "cpu"
    dtype: Literal["float32", "float64"] = "float64"
    calculator_type: Literal["mace_mp", "mace_off", "mace_anicc"] = "mace_mp"
    dispersion: bool = False
    damping: str = "bj"
    dispersion_xc: str = "pbe"
    dispersion_cutoff: float = 21.167088422553647
    type: Literal["ase-mace"] = field(default="ase-mace", init=False)

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("checkpoint must not be empty")
        if self.dtype not in _DTYPES:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.calculator_type not in {
            "mace_mp",
            "mace_off",
            "mace_anicc",
        }:
            raise ValueError(
                f"Unsupported MACE calculator type: {self.calculator_type}"
            )
        _positive_number("dispersion_cutoff", self.dispersion_cutoff)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RootstockConfig:
    """Run a Rootstock-managed checkpoint through its ASE calculator."""

    checkpoint: str
    cluster: str | None = None
    root: str | None = None
    cache_root: str | None = None
    setup_kwargs: dict[str, Any] = field(default_factory=dict)
    timeout: float = 600.0
    weights: str | None = None
    device: str = "cpu"
    type: Literal["rootstock"] = field(default="rootstock", init=False)

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("checkpoint must not be empty")
        if self.cluster is not None and self.root is not None:
            raise ValueError("Rootstock cannot specify both cluster and root")
        _positive_number("timeout", self.timeout)
        try:
            json.dumps(self.setup_kwargs, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "setup_kwargs must contain finite JSON data"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NVAlchemiMACEConfig:
    """Run MACE using NVIDIA ALCHEMI Toolkit."""

    checkpoint: str
    device: str = "cuda"
    dtype: Literal["float32", "float64"] = "float32"
    dt: float = 0.1
    compile_model: bool = False
    enable_cueq: bool = False
    type: Literal["nvalchemi-mace"] = field(
        default="nvalchemi-mace", init=False
    )

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("checkpoint must not be empty")
        if self.dtype not in _DTYPES:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        _positive_number("dt", self.dt)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MLIPBackendConfig: TypeAlias = (
    ASEMACEConfig | RootstockConfig | NVAlchemiMACEConfig
)


@dataclass(frozen=True)
class MLIPCalculationConfig:
    """Calculation settings shared by all MLIP backends."""

    driver: Literal["energy", "opt"] = "energy"
    optimizer: Literal["bfgs", "lbfgs", "gpmin", "fire", "mdmin"] = "fire"
    fmax: float = 0.01
    steps: int = 1000

    def __post_init__(self) -> None:
        if self.driver not in {"energy", "opt"}:
            raise ValueError(f"Unsupported MLIP driver: {self.driver}")
        if self.optimizer not in _OPTIMIZERS:
            raise ValueError(f"Unsupported ASE optimizer: {self.optimizer}")
        _positive_number("fmax", self.fmax)
        _positive_integer("steps", self.steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
