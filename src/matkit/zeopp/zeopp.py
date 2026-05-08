from pathlib import Path
import shutil
import subprocess
import tempfile


VALID_ANALYSES = {"res", "sa", "vol", "psd", "chan"}


def _find_network_binary(network_path: str | None = None) -> str:
    """Locate the Zeo++ network binary.

    Args:
        network_path: Explicit path to the binary. If None, searches PATH.

    Returns:
        Path to the network binary.

    Raises:
        FileNotFoundError: If the binary cannot be found.
    """
    if network_path is not None:
        p = Path(network_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Zeo++ 'network' binary not found at: {network_path}"
            )
        return str(p)

    found = shutil.which("network")
    if found is None:
        raise FileNotFoundError(
            "Zeo++ 'network' binary not found on PATH. "
            "Provide the path via network_path or add it to your PATH."
        )
    return found


def _parse_res(filepath: Path) -> dict:
    """Parse a .res file containing pore diameter results.

    Format: ``structure_name Di Df Dif``

    Args:
        filepath: Path to the .res file.

    Returns:
        Dict with Di, Df, Dif in Angstrom.
    """
    try:
        text = filepath.read_text().strip()
        parts = text.split()
        return {
            "Di": float(parts[1]),
            "Df": float(parts[2]),
            "Dif": float(parts[3]),
            "unit": "Angstrom",
        }
    except Exception as e:
        raise ValueError(f"Failed to parse .res file {filepath}: {e}") from e


def _parse_sa(filepath: Path) -> dict:
    """Parse a .sa file containing accessible surface area results.

    Format is a single line with key: value pairs separated by spaces.

    Args:
        filepath: Path to the .sa file.

    Returns:
        Dict with unitcell_volume, density, ASA, NASA values in various units.
    """
    try:
        text = filepath.read_text().strip()
        tokens = text.split()
        result = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith("Unitcell_volume:"):
                result["unitcell_volume"] = float(tokens[i + 1])
                i += 2
            elif token.startswith("Density:"):
                result["density"] = float(tokens[i + 1])
                i += 2
            elif token == "ASA_A^2:":
                result["ASA"] = float(tokens[i + 1])
                i += 2
            elif token == "ASA_m^2/cm^3:":
                result["ASA_m2_cm3"] = float(tokens[i + 1])
                i += 2
            elif token == "ASA_m^2/g:":
                result["ASA_m2_g"] = float(tokens[i + 1])
                i += 2
            elif token == "NASA_A^2:":
                result["NASA"] = float(tokens[i + 1])
                i += 2
            elif token == "NASA_m^2/cm^3:":
                result["NASA_m2_cm3"] = float(tokens[i + 1])
                i += 2
            elif token == "NASA_m^2/g:":
                result["NASA_m2_g"] = float(tokens[i + 1])
                i += 2
            else:
                i += 1
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse .sa file {filepath}: {e}") from e


def _parse_vol(filepath: Path) -> dict:
    """Parse a .vol file containing accessible volume results.

    Args:
        filepath: Path to the .vol file.

    Returns:
        Dict with unitcell_volume, density, AV, NAV values in various units.
    """
    try:
        text = filepath.read_text().strip()
        tokens = text.split()
        result = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith("Unitcell_volume:"):
                result["unitcell_volume"] = float(tokens[i + 1])
                i += 2
            elif token.startswith("Density:"):
                result["density"] = float(tokens[i + 1])
                i += 2
            elif token == "AV_A^3:":
                result["AV"] = float(tokens[i + 1])
                i += 2
            elif token == "AV_Volume_fraction:":
                result["AV_volume_fraction"] = float(tokens[i + 1])
                i += 2
            elif token == "AV_cm^3/g:":
                result["AV_cm3_g"] = float(tokens[i + 1])
                i += 2
            elif token == "NAV_A^3:":
                result["NAV"] = float(tokens[i + 1])
                i += 2
            elif token == "NAV_Volume_fraction:":
                result["NAV_volume_fraction"] = float(tokens[i + 1])
                i += 2
            elif token == "NAV_cm^3/g:":
                result["NAV_cm3_g"] = float(tokens[i + 1])
                i += 2
            else:
                i += 1
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse .vol file {filepath}: {e}") from e


def _parse_psd(filepath: Path) -> dict:
    """Parse a .psd file containing pore size distribution histogram.

    Args:
        filepath: Path to the .psd file.

    Returns:
        Dict with bin_lower, bin_upper, counts lists and bin_size.
    """
    try:
        lines = filepath.read_text().strip().splitlines()
        bin_lower = []
        bin_upper = []
        counts = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    low = float(parts[0])
                    count = float(parts[1])
                    bin_lower.append(low)
                    counts.append(count)
                except ValueError:
                    continue

        bin_size = bin_lower[1] - bin_lower[0] if len(bin_lower) > 1 else 0.0
        bin_upper = [b + bin_size for b in bin_lower]

        return {
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "counts": counts,
            "bin_size": bin_size,
        }
    except Exception as e:
        raise ValueError(f"Failed to parse .psd file {filepath}: {e}") from e


def _parse_chan(filepath: Path) -> dict:
    """Parse a .chan file containing channel identification results.

    Format: ``structure_name N channels identified of dimensionality d1 d2 ...``

    Args:
        filepath: Path to the .chan file.

    Returns:
        Dict with num_channels and dimensionalities list.
    """
    try:
        text = filepath.read_text().strip()
        parts = text.split()
        channels_idx = parts.index("channels")
        num_channels = int(parts[channels_idx - 1])
        dim_idx = parts.index("dimensionality")
        dimensionalities = [int(d) for d in parts[dim_idx + 1 :]]
        return {
            "num_channels": num_channels,
            "dimensionalities": dimensionalities,
        }
    except Exception as e:
        raise ValueError(f"Failed to parse .chan file {filepath}: {e}") from e


_PARSERS = {
    "res": _parse_res,
    "sa": _parse_sa,
    "vol": _parse_vol,
    "psd": _parse_psd,
    "chan": _parse_chan,
}


def get_output_data(
    output_path: str,
    analyses: list[str] | None = None,
) -> dict:
    """Parse pre-existing Zeo++ output files.

    Args:
        output_path: Path to directory containing Zeo++ output files.
        analyses: Which analyses to parse. If None, auto-detects from
            available files (.res, .sa, .vol, .psd, .chan).

    Returns:
        Dict with 'success' key and per-analysis result sub-dicts.

    Raises:
        FileNotFoundError: If output_path does not exist.
        ValueError: If specified analyses are invalid or parsing fails.
    """
    outdir = Path(output_path)
    if not outdir.exists():
        raise FileNotFoundError(f"Output path does not exist: {output_path}")

    if analyses is not None:
        invalid = set(analyses) - VALID_ANALYSES
        if invalid:
            raise ValueError(
                f"Invalid analysis type(s): {invalid}. Valid: {VALID_ANALYSES}"
            )

    results = {"success": False}

    if outdir.is_file():
        ext = outdir.suffix.lstrip(".")
        if ext in _PARSERS:
            results[ext] = _PARSERS[ext](outdir)
            results["success"] = True
        return results

    # Directory mode: find output files
    detect = analyses if analyses is not None else list(VALID_ANALYSES)
    for analysis in detect:
        matches = list(outdir.glob(f"*.{analysis}"))
        if matches:
            results[analysis] = _PARSERS[analysis](matches[0])

    if len(results) > 1:  # has at least one analysis result beyond 'success'
        results["success"] = True
    return results


def run_zeopp(
    cif: str,
    analyses: list[str] | None = None,
    probe_radius: float = 1.86,
    chan_radius: float = 1.86,
    num_samples: int = 2000,
    ha: bool = True,
    radii_file: str | None = None,
    network_path: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run Zeo++ network binary on a CIF structure file.

    Builds a single command combining all requested analysis flags
    for efficiency (single Voronoi network construction).

    Args:
        cif: Path to the input CIF structure file.
        analyses: List of analysis types to run. Valid values:
            'res', 'sa', 'vol', 'psd', 'chan'.
            Defaults to ['res'] if None.
        probe_radius: Probe molecule radius in Angstrom.
        chan_radius: Channel radius in Angstrom.
        num_samples: Number of Monte Carlo samples for sa/vol/psd.
        ha: Use high accuracy (-ha flag). Defaults to True.
        radii_file: Path to atomic radii file (e.g. UFF.rad).
            Passed via -r flag. Defaults to the bundled
            UFF.rad if None.
        network_path: Explicit path to the network binary.
        output_dir: Directory for output files. Uses a temp directory
            if None.

    Returns:
        Dict with 'success', 'results' (per-analysis sub-dicts), and
        'error' keys.

    Raises:
        FileNotFoundError: If the CIF file, radii file, or network
            binary is missing.
        ValueError: If analysis types are invalid or execution fails.
    """
    cifpath = Path(cif)
    if not cifpath.exists():
        raise FileNotFoundError(f"CIF file does not exist: {cif}")

    if radii_file is None:
        radii_file = str(
            Path(__file__).parent / "files" / "UFF.rad"
        )

    radii_path = Path(radii_file)
    if not radii_path.exists():
        raise FileNotFoundError(
            f"Radii file does not exist: {radii_file}"
        )

    if analyses is None:
        analyses = ["res"]

    invalid = set(analyses) - VALID_ANALYSES
    if invalid:
        raise ValueError(
            f"Invalid analysis type(s): {invalid}. Valid: {VALID_ANALYSES}"
        )

    binary = _find_network_binary(network_path)

    use_temp = output_dir is None
    if use_temp:
        workdir = Path(tempfile.mkdtemp(prefix="zeopp_"))
    else:
        workdir = Path(output_dir)
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy CIF to working directory
        cif_dest = workdir / cifpath.name
        shutil.copy(cifpath, cif_dest)

        # Copy radii file to working directory
        rad_dest = workdir / radii_path.name
        shutil.copy(radii_path, rad_dest)

        # Build command
        cmd = [binary]

        if ha:
            cmd.append("-ha")

        cmd.extend(["-r", str(rad_dest)])

        for analysis in analyses:
            if analysis == "res":
                cmd.extend(["-res"])
            elif analysis == "sa":
                cmd.extend([
                    "-sa", str(probe_radius),
                    str(chan_radius), str(num_samples),
                ])
            elif analysis == "vol":
                cmd.extend([
                    "-vol", str(probe_radius),
                    str(chan_radius), str(num_samples),
                ])
            elif analysis == "psd":
                cmd.extend([
                    "-psd", str(probe_radius),
                    str(chan_radius), str(num_samples),
                ])
            elif analysis == "chan":
                cmd.extend(["-chan", str(probe_radius)])
        cmd.append(str(cif_dest))

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ValueError(
                f"Zeo++ network failed (exit code {proc.returncode}): "
                f"{proc.stderr.strip()}"
            )

        # Parse output files
        result = {"success": False, "results": {}, "error": None}
        stem = cif_dest.stem
        for analysis in analyses:
            out_file = workdir / f"{stem}.{analysis}"
            if out_file.exists():
                result["results"][analysis] = _PARSERS[analysis](out_file)

        result["success"] = len(result["results"]) > 0
        return result

    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise ValueError(f"Zeo++ execution failed: {e}") from e
    finally:
        if use_temp:
            shutil.rmtree(workdir, ignore_errors=True)
