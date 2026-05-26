"""MatKit: A modular Python toolkit for molecular simulations."""

__version__ = "0.1.0"

_SUBMODULES = {
    "graspa",
    "graspa_sycl",
    "raspa2",
    "raspa3",
    "zeopp",
    "mlip",
    "utils",
    "io",
    "plot",
    "tobacco",
    "orca",
    "pacmof2",
}


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"matkit.{name}")
    raise AttributeError(f"module 'matkit' has no attribute {name!r}")
