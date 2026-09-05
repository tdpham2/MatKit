"""Engine syntax and documented output shapes, without an installed Zeo++."""

from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from matkit.api import PoreRequest, StructureRef, run
from matkit.api import adapters
from matkit.zeopp import zeopp


@pytest.mark.parametrize("interface", ["legacy", "unified"])
def test_unequal_radii_and_engine_output_names(
    interface, sample_cif, tmp_path, monkeypatch
):
    commands = []
    data = Path(__file__).parent / "data" / "zeopp"

    def engine(directory, stem, arguments):
        commands.append(arguments)
        for analysis in ("sa", "vol", "psd", "chan"):
            suffix = "psd_histo" if analysis == "psd" else analysis
            shutil.copyfile(
                data / f"test_structure.{analysis}",
                directory / f"{stem}.{suffix}",
            )

    if interface == "legacy":
        monkeypatch.setattr(zeopp, "_find_network_binary", lambda _: "network")

        def launch(command, **kwargs):
            cif = Path(command[-1])
            engine(cif.parent, cif.stem, command[1:])
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(zeopp.subprocess, "run", launch)
        result = zeopp.run_zeopp(
            sample_cif,
            analyses=["sa", "vol", "psd", "chan"],
            probe_radius=1.2,
            chan_radius=1.8,
            num_samples=1234,
            output_dir=str(tmp_path / "legacy"),
        )
        assert result["success"]
        results = result["results"]
    else:

        def launch(root, execution, name, arguments):
            engine(root / "work", "structure", arguments)

        monkeypatch.setattr(adapters, "external_command", launch)
        result = run(
            PoreRequest(
                structure=StructureRef(path=sample_cif),
                analyses=["sa", "vol", "psd", "chan"],
                probe_radius=1.2,
                channel_radius=1.8,
                num_samples=1234,
            ),
            output_dir=tmp_path / "unified",
        )
        assert result.accepted, result.failure
        results = result.payload.results

    for flag in ("-sa", "-vol", "-psd"):
        index = commands[0].index(flag)
        assert commands[0][index + 1 : index + 4] == ["1.8", "1.2", "1234"]
    index = commands[0].index("-chan")
    assert commands[0][index + 1] == "1.2"
    assert results["chan"]["dimensionalities"] == [3, 3]
    assert results["psd"]["counts"][3] == 25


@pytest.mark.parametrize("suffix", ["psd_histo", "psd"])
def test_histogram_discovery_and_explicit_file(suffix, tmp_path):
    path = tmp_path / f"structure.{suffix}"
    path.write_text(
        "# diameter count cumulative derivative\n0.1 2 0.5 1\n0.2 2 1 1\n"
    )
    for target in (path, tmp_path):
        result = zeopp.get_output_data(str(target), analyses=["psd"])
        assert result["success"]
        assert result["psd"]["counts"] == [2, 2]


def test_zero_channels_and_header_only_legacy_file(tmp_path):
    path = tmp_path / "structure.chan"
    for count, dimensions in ((0, []), (1, [2])):
        path.write_text(
            f"structure {count} channels identified of dimensionality "
            + " ".join(map(str, dimensions))
            + "\n"
        )
        assert zeopp.get_output_data(str(path))["chan"] == {
            "num_channels": count,
            "dimensionalities": dimensions,
        }


@pytest.mark.parametrize(
    "contents",
    [
        "structure 2 channels identified of dimensionality 3\n",
        "structure 1 channels identified of dimensionality 4\n",
        "structure 1 channels identified of dimensionality 3\n"
        "Channel 0 1 nan 1\n",
        "structure 1 channels identified of dimensionality 3\nChannel 0 1\n",
        "structure 2 channels identified of dimensionality 3 3\n"
        "Channel 0 1 1 1\n",
    ],
)
def test_malformed_channels_fail(contents, tmp_path):
    path = tmp_path / "structure.chan"
    path.write_text(contents)
    with pytest.raises(ValueError):
        zeopp.get_output_data(str(path))


@pytest.mark.parametrize(
    "contents",
    [
        "# header only\n",
        "0.1\n",
        "0.1 broken\n",
        "0.1 nan\n",
        "0.1 -1\n",
        "0.1 1\n0.1 2\n",
        "0.1 1\n0.2 2\n0.4 1\n",
    ],
)
def test_malformed_histograms_fail(contents, tmp_path):
    path = tmp_path / "structure.psd_histo"
    path.write_text(contents)
    with pytest.raises(ValueError):
        zeopp.get_output_data(str(path))


@pytest.mark.parametrize(
    "analysis,contents",
    [
        ("res", "structure nan 2 3\n"),
        ("sa", "ASA_A^2: 1\n"),
        ("vol", "AV_A^3: 1\n"),
    ],
)
def test_incomplete_or_nonfinite_output_fails(analysis, contents, tmp_path):
    path = tmp_path / f"structure.{analysis}"
    path.write_text(contents)
    with pytest.raises(ValueError):
        zeopp.get_output_data(str(tmp_path), analyses=[analysis])


def test_missing_requested_analysis_fails(sample_cif, tmp_path, monkeypatch):
    monkeypatch.setattr(zeopp, "_find_network_binary", lambda _: "network")
    monkeypatch.setattr(
        zeopp.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stderr=""),
    )
    with pytest.raises(ValueError, match="Missing requested"):
        zeopp.run_zeopp(sample_cif, output_dir=str(tmp_path / "run"))
    with pytest.raises(ValueError, match="Missing requested"):
        zeopp.get_output_data(str(tmp_path), analyses=["res"])


def test_invalid_radius_relation_rejected_by_request(sample_cif):
    with pytest.raises(ValueError, match="must not exceed"):
        PoreRequest(
            structure=StructureRef(path=sample_cif),
            analyses=["sa"],
            probe_radius=1.8,
            channel_radius=1.2,
        )
