"""Parsers for isotherm JSON data files.

Supports two formats produced by gRASPA / RASPA simulations:

1. **Single-component isotherms** -- keys like ``"0.1bar_298K"``
   or ``"314.2Pa_298K"``, each containing uptake/error for one
   adsorbate at one pressure point.

2. **Mixture isotherms (RH-based)** -- keys like ``"50_RH"``,
   each containing ``<name>_uptake`` / ``<name>_error`` pairs
   for every adsorbate in the mixture.
"""

import json
import math
import re
from pathlib import Path

# Regex patterns for key formats
_PRESSURE_BAR_RE = re.compile(r"^(?P<pressure>[\d.]+)bar_(?P<temp>[\d.]+)K$")
_PRESSURE_PA_RE = re.compile(r"^(?P<pressure>[\d.]+)Pa_(?P<temp>[\d.]+)K$")
_RH_RE = re.compile(r"^(?P<rh>\d+)_RH$")


def _safe_float(value):
    """Convert a value to float, returning NaN for invalid values.

    Handles string representations of numbers, ``'-nan'``,
    ``'-inf'``, ``'inf'``, and ``None``.
    """
    if value is None:
        return float("nan")
    try:
        result = float(value)
        if math.isinf(result) or math.isnan(result):
            return float("nan")
        return result
    except (ValueError, TypeError):
        return float("nan")


def detect_format(data: dict) -> str:
    """Detect the isotherm JSON format from its keys.

    Args:
        data: Parsed JSON dictionary.

    Returns:
        ``"single"`` for single-component pressure isotherms,
        ``"mixture_rh"`` for relative-humidity mixture isotherms.

    Raises:
        ValueError: If the format cannot be determined.
    """
    if not data:
        raise ValueError("Empty data dictionary")

    keys = list(data.keys())
    sample = keys[0]

    if _RH_RE.match(sample):
        return "mixture_rh"

    if _PRESSURE_BAR_RE.match(sample) or _PRESSURE_PA_RE.match(sample):
        return "single"

    raise ValueError(
        f"Cannot detect isotherm format from key: '{sample}'. "
        "Expected patterns like '0.1bar_298K', '314.2Pa_298K', "
        "or '50_RH'."
    )


def parse_single_isotherm(data: dict) -> dict:
    """Parse a single-component isotherm JSON dict.

    Args:
        data: Dict with keys like ``"0.1bar_298K"`` mapping to
            dicts with ``uptake``, ``error``, and optional ``qst``
            fields.

    Returns:
        Dict with keys:
        - ``format``: ``"single"``
        - ``pressure_unit``: ``"bar"`` or ``"Pa"``
        - ``pressures``: sorted list of float pressures
        - ``uptakes``: list of float uptakes (sorted by pressure)
        - ``errors``: list of float errors (sorted by pressure)
        - ``unit``: uptake unit string (e.g. ``"mol/kg"``)
        - ``temperature``: float temperature in K (from first key)
        - ``qst``: list of float QST values (NaN if absent)
        - ``qst_errors``: list of float QST errors (NaN if absent)
        - ``qst_unit``: QST unit string or ``None``
    """
    entries = []
    pressure_unit = None

    for key, values in data.items():
        m_bar = _PRESSURE_BAR_RE.match(key)
        m_pa = _PRESSURE_PA_RE.match(key)

        if m_bar:
            pressure = float(m_bar.group("pressure"))
            temperature = float(m_bar.group("temp"))
            pressure_unit = pressure_unit or "bar"
        elif m_pa:
            pressure = float(m_pa.group("pressure"))
            temperature = float(m_pa.group("temp"))
            pressure_unit = pressure_unit or "Pa"
        else:
            continue  # skip unrecognised keys

        entries.append(
            {
                "pressure": pressure,
                "temperature": temperature,
                "uptake": _safe_float(values.get("uptake")),
                "error": _safe_float(values.get("error")),
                "unit": values.get("unit", "mol/kg"),
                "qst": _safe_float(values.get("qst")),
                "qst_error": _safe_float(values.get("error_qst")),
                "qst_unit": values.get("qst_unit"),
            }
        )

    entries.sort(key=lambda e: e["pressure"])

    return {
        "format": "single",
        "pressure_unit": pressure_unit or "bar",
        "pressures": [e["pressure"] for e in entries],
        "uptakes": [e["uptake"] for e in entries],
        "errors": [e["error"] for e in entries],
        "unit": entries[0]["unit"] if entries else "mol/kg",
        "temperature": entries[0]["temperature"] if entries else None,
        "qst": [e["qst"] for e in entries],
        "qst_errors": [e["qst_error"] for e in entries],
        "qst_unit": (entries[0]["qst_unit"] if entries else None),
    }


def _discover_adsorbates(data: dict) -> list[str]:
    """Discover adsorbate names from mixture isotherm keys.

    Looks for keys matching the pattern ``<name>_uptake`` in the
    first data point and returns a sorted list of names.
    """
    first_key = next(iter(data))
    point = data[first_key]

    adsorbates = []
    for k in point:
        if k.endswith("_uptake"):
            name = k[: -len("_uptake")]
            adsorbates.append(name)

    return sorted(adsorbates)


def _discover_selectivity_keys(data: dict) -> list[str]:
    """Discover selectivity field names from mixture isotherm data.

    Looks for keys matching ``*_selectivity`` in the first point.
    """
    first_key = next(iter(data))
    point = data[first_key]
    return sorted(k for k in point if k.endswith("_selectivity"))


def parse_mixture_isotherm(data: dict) -> dict:
    """Parse a mixture (RH-based) isotherm JSON dict.

    Adsorbate names are auto-discovered from the data keys
    (e.g. ``co2_uptake`` → adsorbate ``"co2"``).

    Args:
        data: Dict with keys like ``"50_RH"`` mapping to dicts
            containing ``<name>_uptake`` and ``<name>_error``
            fields.

    Returns:
        Dict with keys:
        - ``format``: ``"mixture_rh"``
        - ``rh_values``: sorted list of int RH percentages
        - ``adsorbates``: sorted list of adsorbate name strings
        - ``uptakes``: dict mapping adsorbate name → list of floats
        - ``errors``: dict mapping adsorbate name → list of floats
        - ``selectivity``: dict mapping selectivity key → list of
          floats (e.g. ``"co2_n2_selectivity"`` → ``[...]``)
    """
    adsorbates = _discover_adsorbates(data)
    selectivity_keys = _discover_selectivity_keys(data)

    entries = []
    for key, values in data.items():
        m = _RH_RE.match(key)
        if not m:
            continue
        rh = int(m.group("rh"))
        entries.append({"rh": rh, "values": values})

    entries.sort(key=lambda e: e["rh"])

    result = {
        "format": "mixture_rh",
        "rh_values": [e["rh"] for e in entries],
        "adsorbates": adsorbates,
        "uptakes": {},
        "errors": {},
        "selectivity": {},
    }

    for ads in adsorbates:
        result["uptakes"][ads] = [
            _safe_float(e["values"].get(f"{ads}_uptake")) for e in entries
        ]
        result["errors"][ads] = [
            _safe_float(e["values"].get(f"{ads}_error")) for e in entries
        ]

    for sel_key in selectivity_keys:
        result["selectivity"][sel_key] = [
            _safe_float(e["values"].get(sel_key)) for e in entries
        ]

    return result


def collect_data_files(
    data: tuple[str, ...] | list[str] = (),
    data_dir: str | None = None,
    pattern: str = "*.json",
) -> list[str]:
    """Build a sorted list of data file paths.

    Combines explicitly listed files with all files matching
    *pattern* inside an optional directory.  Files from the
    directory are sorted alphabetically so the order is
    deterministic.

    Args:
        data: Explicit file paths.
        data_dir: Optional directory to scan for JSON files.
        pattern: Glob pattern for directory scan (default
            ``"*.json"``).

    Returns:
        Deduplicated list of resolved file path strings.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
        ValueError: If no files are found.
    """
    paths: list[Path] = []

    for p in data:
        paths.append(Path(p).resolve())

    if data_dir is not None:
        d = Path(data_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        paths.extend(sorted(d.glob(pattern)))

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(str(p))

    if not unique:
        raise ValueError(
            "No data files found. Provide --data and/or --data-dir."
        )

    return unique


def load_isotherm(filepath: str) -> dict:
    """Load and parse an isotherm JSON file.

    Auto-detects the format (single-component or mixture) and
    returns a normalised data structure.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed isotherm dict (see :func:`parse_single_isotherm`
        or :func:`parse_mixture_isotherm` for structure).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format cannot be detected or parsed.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, "r") as f:
        data = json.load(f)

    fmt = detect_format(data)
    if fmt == "single":
        return parse_single_isotherm(data)
    elif fmt == "mixture_rh":
        return parse_mixture_isotherm(data)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
