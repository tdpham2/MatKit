"""Run these checks from an installed wheel as well as editable checkouts."""

from importlib import resources
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "package,path",
    [
        ("graspa", "files/template/simulation.input"),
        ("graspa", "files/template_mixture_isotherm/template_mixture/SO2.def"),
        ("pygraspa", "files/template_mixture_isotherm/template_mixture/H2.def"),
        ("graspa_sycl", "files/template/simulation.input"),
        ("raspa2", "files/template/simulation.input"),
        ("raspa3", "files/template/CO2.def"),
        ("zeopp", "files/UFF.rad"),
    ],
)
def test_bundled_resources(package, path):
    assert resources.files(f"matkit.{package}").joinpath(path).read_bytes()


def test_core_import_keeps_optional_runtimes_unloaded():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import matkit.api; "
            "assert not {'torch', 'mace', 'rootstock', 'nvalchemi', 'mcp'} "
            "& sys.modules.keys()",
        ],
        check=True,
    )


def test_installed_location():
    if os.environ.get("MATKIT_WHEEL_TEST") == "1":
        import matkit

        assert "site-packages" in Path(matkit.__file__).parts
