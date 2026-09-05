"""Versioned scientific requests and portable, engine-independent results."""

from __future__ import annotations

import json
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    TypeAdapter,
    model_validator,
)

Positive = Annotated[float, Field(gt=0, strict=True)]
Nonnegative = Annotated[float, Field(ge=0, strict=True)]
Count = Annotated[StrictInt, Field(ge=1)]
Cycles = Annotated[StrictInt, Field(ge=0)]
Vector = tuple[float, float, float]
Matrix = tuple[Vector, Vector, Vector]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        revalidate_instances="always",
    )

    @model_validator(mode="after")
    def finite_json(self):
        # Reject non-standard JSON in nested JsonValue fields too.
        json.dumps(self.model_dump(mode="json"), allow_nan=False)
        return self


class StructureData(Model):
    atomic_numbers: list[Count]
    positions: list[Vector]
    cell: Matrix
    pbc: tuple[bool, bool, bool]
    atom_ids: list[str]
    arrays: dict[str, JsonValue] = Field(default_factory=dict)
    info: dict[str, JsonValue] = Field(default_factory=dict)
    constraints: list[dict[str, JsonValue]] = Field(default_factory=list)
    labels: list[str] | None = None
    bonds: list[tuple[str, str]] = Field(default_factory=list)
    parent_sha256: Digest | None = None
    derived_from: dict[str, Digest] = Field(default_factory=dict)

    @model_validator(mode="after")
    def atom_correspondence(self):
        size = len(self.atomic_numbers)
        if (
            not size
            or len(self.positions) != size
            or len(self.atom_ids) != size
        ):
            raise ValueError(
                "Species, positions and atom_ids require equal nonzero lengths"
            )
        if len(set(self.atom_ids)) != size:
            raise ValueError("atom_ids must be unique")
        if self.labels is not None and len(self.labels) != size:
            raise ValueError("labels must correspond to every atom")
        for name, array in self.arrays.items():
            if not isinstance(array, list) or len(array) != size:
                raise ValueError(f"Array {name} must correspond to every atom")
        if any(
            a not in self.atom_ids or b not in self.atom_ids
            for a, b in self.bonds
        ):
            raise ValueError("Bond endpoints must refer to atom_ids")
        return self


class StructureRef(Model):
    path: str
    sha256: Digest | None = None
    metadata: StructureData | None = None


class ExecutionConfig(Model):
    """Execution location belongs to the caller, not the scientific method."""

    mode: Literal["inprocess", "subprocess"] = "inprocess"
    python: str | None = None
    device: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    executables: dict[Literal["zeopp", "graspa"], list[str]] = Field(
        default_factory=dict
    )
    timeout_s: Positive | None = None

    @model_validator(mode="after")
    def commands_not_empty(self):
        if any(
            not command or any(not token for token in command)
            for command in self.executables.values()
        ):
            raise ValueError(
                "Executable commands must be nonempty argument lists"
            )
        return self


class MLIPMethod(Model):
    checkpoint: str = Field(min_length=1)
    calculator_type: Literal["mace_mp", "mace_off", "mace_anicc"] = "mace_mp"
    dispersion: bool = False
    damping: str = "bj"
    dispersion_xc: str = "pbe"
    dispersion_cutoff: Positive = 21.167088422553647


class MACEAdapter(Model):
    type: Literal["ase-mace"] = "ase-mace"
    dtype: Literal["float32", "float64"] | None = None


class RootstockAdapter(Model):
    type: Literal["rootstock"] = "rootstock"
    cluster: str | None = None
    root: str | None = None
    cache_root: str | None = None
    setup_kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    timeout: Positive = 600
    weights: str | None = None

    @model_validator(mode="after")
    def location(self):
        if self.cluster is not None and self.root is not None:
            raise ValueError("Rootstock cannot specify both cluster and root")
        return self


class AlchemiAdapter(Model):
    type: Literal["nvalchemi-mace"] = "nvalchemi-mace"
    dtype: Literal["float32", "float64"] = "float32"
    dt: Positive | None = None
    compile_model: bool = False
    enable_cueq: bool = False
    batch_size: Count = 16
    max_atoms: Count | None = None


CalculatorAdapter = Annotated[
    Union[MACEAdapter, RootstockAdapter, AlchemiAdapter],
    Field(discriminator="type"),
]


class RequestBase(Model):
    schema_name: Literal["matkit.request"] = "matkit.request"
    schema_version: Literal[1] = 1
    structure: StructureRef


class CalculatorRequest(RequestBase):
    method: MLIPMethod
    adapter: CalculatorAdapter = Field(default_factory=MACEAdapter)

    @model_validator(mode="after")
    def applicable_method(self):
        method = self.method
        defaults = MLIPMethod(checkpoint=method.checkpoint)
        if self.adapter.type != "ase-mace" and method != defaults:
            raise ValueError(
                "MACE factory/dispersion settings require ase-mace"
            )
        if method.dispersion and method.calculator_type != "mace_mp":
            raise ValueError("Dispersion requires mace_mp")
        if (
            isinstance(self.adapter, MACEAdapter)
            and method.calculator_type == "mace_anicc"
            and self.adapter.dtype is not None
        ):
            raise ValueError("mace_anicc controls its own precision")
        return self


class EvaluateRequest(CalculatorRequest):
    operation: Literal["evaluate"] = "evaluate"
    properties: list[Literal["potential_energy", "forces", "stress"]] = Field(
        default_factory=lambda: ["potential_energy"], min_length=1
    )

    @model_validator(mode="after")
    def applicable_options(self):
        if len(set(self.properties)) != len(self.properties):
            raise ValueError("Requested properties must be unique")
        if (
            isinstance(self.adapter, AlchemiAdapter)
            and self.adapter.dt is not None
        ):
            raise ValueError("ALCHEMI dt requires relaxation")
        return self


class RelaxRequest(CalculatorRequest):
    operation: Literal["relax"] = "relax"
    optimizer: Literal["bfgs", "lbfgs", "gpmin", "fire", "mdmin"] = "fire"
    fmax: Positive = 0.01
    steps: Count = 1000

    @model_validator(mode="after")
    def native_optimizer(self):
        if self.adapter.type == "nvalchemi-mace" and self.optimizer != "fire":
            raise ValueError("ALCHEMI supports only FIRE")
        return self


class PoreRequest(RequestBase):
    operation: Literal["pores"] = "pores"
    adapter: Literal["zeopp"] = "zeopp"
    analyses: list[Literal["res", "sa", "vol", "psd", "chan"]] = Field(
        default_factory=lambda: ["res"], min_length=1
    )
    probe_radius: Positive = 1.86
    channel_radius: Positive = 1.86
    num_samples: Count = 2000
    high_accuracy: bool = True
    radii_file: str | None = None

    @model_validator(mode="after")
    def unique_analyses(self):
        if len(set(self.analyses)) != len(self.analyses):
            raise ValueError("Analyses must be unique")
        if {"sa", "vol", "psd"}.intersection(
            self.analyses
        ) and self.probe_radius > self.channel_radius:
            raise ValueError("probe_radius must not exceed channel_radius")
        return self


class AdsorptionRequest(RequestBase):
    operation: Literal["adsorption"] = "adsorption"
    adapter: Literal["graspa"] = "graspa"
    adsorbate: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    temperature_K: Positive
    pressure_Pa: Nonnegative
    cutoff_angstrom: Positive = 12.8
    initialization_cycles: Cycles = 1000
    equilibration_cycles: Cycles = 0
    production_cycles: Count = 1000
    number_of_blocks: Count = 1
    fugacity_coefficient: Union[Positive, Literal["PR-EOS"]] = "PR-EOS"
    unit: Literal["mol/kg", "mg/g", "g/L"] = "mol/kg"
    template_dir: str | None = None
    net_charge: float = 0


RunRequest = Annotated[
    Union[EvaluateRequest, RelaxRequest, PoreRequest, AdsorptionRequest],
    Field(discriminator="operation"),
]
REQUEST_ADAPTER = TypeAdapter(RunRequest)


def parse_request(value: RunRequest | dict) -> RunRequest:
    return REQUEST_ADAPTER.validate_python(value)


class Artifact(Model):
    path: str
    sha256: Digest
    size_bytes: Annotated[StrictInt, Field(ge=0)]
    media_type: str = "application/octet-stream"
    role: str


class ScientificCheck(Model):
    name: str
    status: Literal["passed", "failed", "unknown", "not_applicable"]
    required: bool = False
    detail: str = ""


class Failure(Model):
    code: str
    stage: str
    message: str


class EvaluationPayload(Model):
    kind: Literal["evaluation"] = "evaluation"
    potential_energy: float | None = None
    energy_unit: Literal["eV"] = "eV"
    energy_definition: str = (
        "calculator potential energy; model-specific reference"
    )
    forces: list[Vector] | None = None
    force_unit: Literal["eV/angstrom"] = "eV/angstrom"
    stress: Matrix | None = None
    stress_unit: Literal["eV/angstrom^3"] = "eV/angstrom^3"
    stress_convention: str = "ASE Cartesian tensor, positive in tension"
    converged: bool | None = None
    n_steps: Cycles | None = None
    final_structure: str | None = None


class PorePayload(Model):
    kind: Literal["pores"] = "pores"
    results: dict[str, dict[str, JsonValue]]


class AdsorptionPayload(Model):
    kind: Literal["adsorption"] = "adsorption"
    component: str
    uptake: float
    uptake_basis: Literal["absolute, per framework mass/volume"] = (
        "absolute, per framework mass/volume"
    )
    unit: Literal["mol/kg", "mg/g", "g/L"]
    uncertainty: Nonnegative
    uncertainty_method: str = "unknown; reported by engine"
    heat_of_adsorption: float
    heat_unit: Literal["kJ/mol"] = "kJ/mol"
    heat_uncertainty: Nonnegative
    heat_convention: str = "as reported by gRASPA"


Payload = Annotated[
    Union[EvaluationPayload, PorePayload, AdsorptionPayload],
    Field(discriminator="kind"),
]


class RunResult(Model):
    schema_name: Literal["matkit.run"] = "matkit.run"
    schema_version: Literal[1] = 1
    run_id: str
    operation: Literal["evaluate", "relax", "pores", "adsorption"]
    state: Literal["prepared", "running", "completed", "failed", "interrupted"]
    numerical_validity: Literal["valid", "invalid", "unknown"] = "unknown"
    checks: list[ScientificCheck] = Field(default_factory=list)
    requested: dict[str, JsonValue]
    resolved: dict[str, JsonValue] = Field(default_factory=dict)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    timings: dict[str, Nonnegative] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    payload: Payload | None = None
    failure: Failure | None = None

    @property
    def accepted(self) -> bool:
        return (
            self.state == "completed"
            and self.numerical_validity == "valid"
            and all(
                not check.required
                or check.status in {"passed", "not_applicable"}
                for check in self.checks
            )
        )

    @model_validator(mode="after")
    def consistent_outcome(self):
        if self.state == "completed" and (
            self.payload is None
            or self.numerical_validity != "valid"
            or self.failure is not None
        ):
            raise ValueError(
                "Completed runs require valid results and no execution failure"
            )
        if self.state in {"failed", "interrupted"} and self.failure is None:
            raise ValueError("Failed/interrupted runs require a failure record")
        if self.payload is not None:
            kind = (
                "evaluation"
                if self.operation in {"evaluate", "relax"}
                else self.operation
            )
            if self.payload.kind != kind:
                raise ValueError("Payload kind does not match operation")
        return self


class BatchResult(Model):
    schema_name: Literal["matkit.batch"] = "matkit.batch"
    schema_version: Literal[1] = 1
    state: Literal["running", "completed", "partial", "failed", "interrupted"]
    items: list[dict[str, JsonValue]]
    failure: Failure | None = None

    @property
    def accepted(self) -> bool:
        return (
            self.state == "completed"
            and self.failure is None
            and bool(self.items)
            and all(item.get("accepted") for item in self.items)
        )
