"""Configuration objects for runtime-selectable MLIP calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias


_DTYPES = {"float32", "float64"}
_OPTIMIZERS = {"bfgs", "lbfgs", "gpmin", "fire", "mdmin"}


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
        if self.dispersion_cutoff <= 0:
            raise ValueError("dispersion_cutoff must be positive")

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
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

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
        if self.dt <= 0:
            raise ValueError("dt must be positive")

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
        if self.fmax <= 0:
            raise ValueError("fmax must be positive")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
