"""Run the usage examples through public APIs and supervised test engines."""

import importlib.util
import json
import os
from pathlib import Path
import sys

from ase import Atoms
from ase.io import write
import pytest

from matkit.api import inspect_run


@pytest.fixture
def example():
    def load(name):
        path = Path(__file__).parents[1] / "examples" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"example_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def profile(tmp_path):
    def create(engine, *arguments):
        path = tmp_path / f"{engine}-execution.json"
        fixture = Path(__file__).parent / "fixtures" / "fake_engine.py"
        path.write_text(
            json.dumps(
                {
                    "executables": {
                        engine: [
                            sys.executable,
                            str(fixture),
                            engine,
                            *arguments,
                        ]
                    }
                }
            )
        )
        return str(path)

    return create


@pytest.fixture
def charged_cif(sample_cif, tmp_path):
    # Synthetic charges belong only to test fixtures, not the usage programs.
    path = tmp_path / "charged.cif"
    lines = []
    for line in Path(sample_cif).read_text().splitlines():
        if line.strip() == "_atom_site_occupancy":
            lines.append("  _atom_site_charge")
        fields = line.split()
        if fields and fields[0] in {"Si", "O"}:
            fields.insert(-1, "0.0")
            line = " ".join(fields)
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.mark.parametrize(
    "name", ["zeopp", "graspa", "graspa_mixture_setup", "mlip"]
)
def test_help_requires_no_optional_engines(example, name, capsys):
    with pytest.raises(SystemExit) as exc:
        example(name).main(["--help"])
    assert exc.value.code == 0
    assert "--outdir" in capsys.readouterr().out


@pytest.mark.parametrize("count", [1, 2])
def test_zeopp_all_analyses_and_batches(
    example, profile, sample_cif, tmp_path, capsys, count
):
    root = tmp_path / "pores"
    code = example("zeopp").main(
        [sample_cif] * count
        + ["--outdir", str(root), "--execution", profile("zeopp")]
    )
    assert code == 0
    stdout = json.loads(capsys.readouterr().out)
    if count == 2:
        assert [item["index"] for item in stdout["items"]] == [0, 1]
        assert all(item["accepted"] for item in stdout["items"])
    assert (root / "worker.stdout.log").is_file()
    for index in range(count):
        bundle = root if count == 1 else root / f"{index:05d}"
        result = inspect_run(bundle)
        assert result.accepted
        assert set(result.payload.results) == {
            "res",
            "sa",
            "vol",
            "psd",
            "chan",
        }
        assert (bundle / "work" / "structure.psd_histo").is_file()


@pytest.mark.parametrize("failure", ["partial", "missing_executable"])
def test_zeopp_failures_are_persisted_and_nonzero(
    example, profile, sample_cif, tmp_path, failure
):
    root = tmp_path / "pores"
    path = Path(profile("zeopp", "--partial"))
    if failure == "missing_executable":
        path.write_text(
            json.dumps({"executables": {"zeopp": [str(tmp_path / "missing")]}})
        )
    assert (
        example("zeopp").main(
            [sample_cif, "--outdir", str(root), "--execution", str(path)]
        )
        == 1
    )
    result = inspect_run(root)
    assert result.state == "failed"
    assert not result.accepted
    assert result.failure is not None


def test_graspa_prepares_distinct_pressure_bundles(
    example, charged_cif, tmp_path, capsys
):
    root = tmp_path / "prepared"
    arguments = [
        charged_cif,
        "--outdir",
        str(root),
        "--prepare-only",
        "--pressure-pa",
        "100000.1",
        "--pressure-pa",
        "100000.2",
    ]
    assert example("graspa").main(arguments) == 0
    records = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]
    assert [record["state"] for record in records] == ["prepared", "prepared"]
    for index, pressure in enumerate([100000.1, 100000.2]):
        bundle = root / f"{index:05d}"
        record = inspect_run(bundle)
        assert record.requested["pressure_Pa"] == pressure
        assert (
            f"Pressure {pressure}"
            in (bundle / "work/simulation.input").read_text()
        )
        assert (bundle / "inputs/structure.cif").read_bytes() == Path(
            charged_cif
        ).read_bytes()
        assert not (bundle / "worker.stdout.log").exists()
    # A rerun must not silently replace the prepared cases.
    assert example("graspa").main(arguments) == 1
    assert inspect_run(root / "00000").run_id == records[0]["run_id"]


@pytest.mark.parametrize("fails", [False, True])
def test_graspa_executes_every_pressure_and_reports_outcomes(
    example, profile, charged_cif, tmp_path, capsys, fails
):
    root = tmp_path / "sweep"
    execution = profile("graspa", *(["--fail"] if fails else []))
    code = example("graspa").main(
        [
            charged_cif,
            "--outdir",
            str(root),
            "--pressure-pa",
            "1000",
            "--pressure-pa",
            "100000",
            "--execution",
            execution,
        ]
    )
    assert code == (1 if fails else 0)
    records = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]
    assert len(records) == 2
    for index in range(2):
        result = inspect_run(root / f"{index:05d}")
        assert result.accepted is not fails
        if fails:
            assert "code 7" in result.failure.message
        else:
            assert result.payload.uptake == 12
            assert result.payload.unit == "mol/kg"


def test_graspa_does_not_invent_charges(example, sample_cif, tmp_path):
    root = tmp_path / "uncharged"
    assert (
        example("graspa").main(
            [sample_cif, "--outdir", str(root), "--prepare-only"]
        )
        == 1
    )
    assert "_atom_site_charge" in inspect_run(root / "00000").failure.message


def test_mixture_preserves_input_and_explicit_composition(
    example, charged_cif, tmp_path
):
    root = tmp_path / "mixture"
    arguments = [charged_cif, "--outdir", str(root)]
    module = example("graspa_mixture_setup")
    assert module.main(arguments) == 0
    content = (root / "simulation.input").read_text()
    assert content.count("MoleculeName") == 2
    assert "CO2" in content and "N2" in content
    assert [
        line.split()[-1]
        for line in content.splitlines()
        if "MolFraction" in line
    ] == ["0.15", "0.85"]
    assert (root / "charged.cif").read_bytes() == Path(charged_cif).read_bytes()
    assert not (root / "run.json").exists()
    assert module.main(arguments) == 1
    assert (root / "simulation.input").read_text() == content


@pytest.fixture
def mlip_inputs(tmp_path):
    # A worker-local MACE factory double exercises the actual subprocess API.
    # No example code or source calculator implementation is patched.
    packages = tmp_path / "engine_packages"
    package = packages / "mace"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "calculators.py").write_text(
        "from ase.calculators.emt import EMT\n"
        "def mace_mp(**kwargs):\n"
        "    return EMT()\n"
    )
    profile = tmp_path / "mlip-execution.json"
    profile.write_text(
        json.dumps(
            {
                "device": "cpu",
                "environment": {
                    "PYTHONPATH": os.pathsep.join([str(packages), *sys.path])
                },
            }
        )
    )
    source = tmp_path / "copper.extxyz"
    write(
        source,
        Atoms("Cu2", positions=[[0, 0, 0], [3, 0, 0]], cell=[10] * 3, pbc=True),
    )
    return str(source), str(profile)


@pytest.mark.parametrize(
    "driver,count", [("energy", 1), ("energy", 2), ("relax", 1)]
)
def test_mlip_worker_results_and_unconverged_relaxation(
    example, mlip_inputs, tmp_path, capsys, driver, count
):
    source, profile = mlip_inputs
    root = tmp_path / "mlip"
    arguments = [source] * count + [
        "--outdir",
        str(root),
        "--backend",
        "ase-mace",
        "--checkpoint",
        "fixture",
        "--execution",
        profile,
        "--driver",
        driver,
    ]
    if driver == "relax":
        arguments += ["--steps", "1", "--fmax", "1e-12"]
    code = example("mlip").main(arguments)
    assert code == (1 if driver == "relax" else 0)
    stdout = json.loads(capsys.readouterr().out)
    if count == 2:
        assert [item["index"] for item in stdout["items"]] == [0, 1]
        assert all(item["accepted"] for item in stdout["items"])
    for index in range(count):
        bundle = root if count == 1 else root / f"{index:05d}"
        result = inspect_run(bundle)
        assert result.state == "completed"
        assert result.payload.potential_energy is not None
        assert len(result.payload.forces) == 2
        assert (bundle / result.payload.final_structure).exists()
        assert result.accepted is (driver == "energy")
        if driver == "relax":
            assert result.payload.converged is False


@pytest.mark.parametrize(
    "options",
    [
        ["--backend", "rootstock", "--dtype", "float32"],
        ["--backend", "ase-mace", "--batch-size", "16"],
        ["--backend", "nvalchemi-mace", "--batch-size", "0"],
        ["--backend", "ase-mace", "--steps", "100"],
    ],
)
def test_invalid_mlip_options_fail_before_preparation(
    example, tmp_path, options
):
    root = tmp_path / "invalid"
    with pytest.raises(SystemExit) as exc:
        example("mlip").main(
            [
                "input.extxyz",
                "--outdir",
                str(root),
                "--checkpoint",
                "medium",
                *options,
            ]
        )
    assert exc.value.code == 2
    assert not root.exists()
