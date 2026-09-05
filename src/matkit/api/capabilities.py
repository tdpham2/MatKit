"""Discovery must not load optional calculators, models or GPU runtimes."""

from importlib.metadata import PackageNotFoundError, version
import shutil


_CAPABILITIES = {
    "ase-mace": (
        "mace-torch",
        ["evaluate", "relax"],
        ["potential_energy", "forces", "stress (model-dependent)"],
        "CPU/CUDA; MACE-supported Python and dependencies",
        "Species, periodicity and supported properties depend on the model",
    ),
    "rootstock": (
        "rootstock",
        ["evaluate", "relax"],
        ["potential_energy", "forces", "stress (deployment-dependent)"],
        "Separately installed Rootstock model environment",
        "Properties and species depend on the deployed model; "
        "caller discovery does not verify the worker environment",
    ),
    "nvalchemi-mace": (
        "nvalchemi-toolkit",
        ["evaluate", "relax"],
        ["potential_energy", "forces", "stress (model-dependent)"],
        "Compatible Python/CUDA stack; native batches; FIRE only",
        "Model-dependent properties/species; unsupported atom arrays, "
        "bonds and constraints are rejected",
    ),
    "zeopp": (
        None,
        ["pores"],
        ["res", "sa", "vol", "psd", "chan"],
        "Zeo++ network executable",
        "Fully periodic structures; explicit radii and requested analyses",
    ),
    "graspa": (
        None,
        ["adsorption"],
        ["single-component absolute uptake", "heat of adsorption"],
        "gRASPA CUDA; charged periodic CIF and force-field definitions",
        "Single component; atom-mapped CIF charges; sampling quality unknown",
    ),
}


def list_capabilities() -> list[dict]:
    capabilities = []
    for adapter, (
        package,
        operations,
        properties,
        environment,
        restrictions,
    ) in _CAPABILITIES.items():
        installed_version = None
        if package:
            try:
                installed_version = version(package)
            except PackageNotFoundError:
                pass
            available = installed_version is not None
        else:
            available = (
                shutil.which("network" if adapter == "zeopp" else "simulate")
                is not None
            )
        capabilities.append(
            {
                "adapter": adapter,
                "operations": operations,
                "properties": properties,
                "available_in_caller": available,
                "installed_version": installed_version,
                "environment": environment,
                "status": "experimental",
                "evidence": "CPU fixtures; real-execution evidence not bundled",
                "geometry_optimization": adapter
                in {"ase-mace", "rootstock", "nvalchemi-mace"},
                "cell_optimization": False,
                "restart": False,
                "native_batch": adapter == "nvalchemi-mace",
                "restrictions": restrictions,
            }
        )
    return capabilities
