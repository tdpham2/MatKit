"""Isotherm plotting functions.

Provides functions for plotting single-component isotherms
(uptake vs pressure), mixture isotherms (uptake vs relative
humidity), and selectivity plots from gRASPA / RASPA simulation
output JSON files.

All functions accept one or more data files for overlay
comparison and offer extensive customisation of labels, scales,
figure size, and output format.
"""

from pathlib import Path

from matkit.plot.parsers import load_isotherm

# Mapping of lowercase adsorbate names to pretty LaTeX-style
# labels used by matplotlib's mathtext renderer.
_PRETTY_LABELS = {
    "co2": r"CO$_2$",
    "n2": r"N$_2$",
    "h2o": r"H$_2$O",
    "h2": r"H$_2$",
    "ch4": r"CH$_4$",
    "so2": r"SO$_2$",
    "o2": r"O$_2$",
    "ar": "Ar",
    "he": "He",
    "kr": "Kr",
    "xe": "Xe",
    "nh3": r"NH$_3$",
    "h2s": r"H$_2$S",
    "no2": r"NO$_2$",
    "co": "CO",
}

# Default marker cycle for multi-series plots
_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]


def _pretty(name: str) -> str:
    """Return a publication-quality label for an adsorbate name."""
    return _PRETTY_LABELS.get(name.lower(), name)


def _get_matplotlib():
    """Import matplotlib, raising a helpful error if missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matkit[plot]"
        )


def plot_single_isotherm(
    data_files: list[str],
    output: str = "isotherm_plot.png",
    dpi: int = 600,
    figsize: tuple[float, float] = (8, 6),
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    labels: list[str] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    no_errorbars: bool = False,
    fontsize_label: int = 24,
    fontsize_tick: int = 16,
    fontsize_legend: int = 16,
) -> str:
    """Plot single-component isotherms (uptake vs pressure).

    Supports overlaying multiple data files on the same axes for
    comparison (e.g. different MOFs at the same conditions).

    Args:
        data_files: List of paths to isotherm JSON files.
        output: Output image file path.
        dpi: Image resolution in dots per inch.
        figsize: Figure dimensions ``(width, height)`` in inches.
        xlabel: Custom x-axis label. Defaults to
            ``"Pressure (<unit>)"``.
        ylabel: Custom y-axis label. Defaults to
            ``"Uptake (<unit>)"``.
        title: Plot title. ``None`` for no title.
        labels: Legend labels, one per data file. Defaults to
            filenames.
        log_x: Use logarithmic x-axis.
        log_y: Use logarithmic y-axis.
        no_errorbars: If ``True``, omit error bars.
        fontsize_label: Font size for axis labels.
        fontsize_tick: Font size for tick labels.
        fontsize_legend: Font size for legend entries.

    Returns:
        The absolute path to the saved plot image.

    Raises:
        ImportError: If matplotlib is not installed.
        FileNotFoundError: If any data file is missing.
        ValueError: If a data file is not single-component format.
    """
    plt = _get_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Pre-parse all files so we can auto-generate labels
    parsed_list: list[dict] = []
    for fpath in data_files:
        parsed = load_isotherm(fpath)
        if parsed["format"] != "single":
            raise ValueError(
                f"Expected single-component format, got "
                f"'{parsed['format']}' in {fpath}"
            )
        parsed_list.append(parsed)

    # Auto-generate labels from temperature when the user
    # did not supply custom labels and multiple files are
    # loaded (typical use-case: same adsorbate at different T).
    auto_labels = labels
    if not labels and len(data_files) > 1:
        temps = [p.get("temperature") for p in parsed_list]
        if all(t is not None for t in temps):
            auto_labels = [
                f"{int(t)} K" if t == int(t) else f"{t} K" for t in temps
            ]

    for i, (fpath, parsed) in enumerate(zip(data_files, parsed_list)):
        label = (
            auto_labels[i]
            if auto_labels and i < len(auto_labels)
            else Path(fpath).stem
        )
        marker = _MARKERS[i % len(_MARKERS)]

        if no_errorbars:
            ax.plot(
                parsed["pressures"],
                parsed["uptakes"],
                f"{marker}-",
                label=label,
            )
        else:
            ax.errorbar(
                parsed["pressures"],
                parsed["uptakes"],
                yerr=parsed["errors"],
                fmt=f"{marker}-",
                label=label,
                capsize=5,
            )

    # Axis labels (use last parsed for metadata)
    pressure_unit = parsed_list[-1]["pressure_unit"]
    uptake_unit = parsed_list[-1].get("unit", "mol/kg")
    ax.set_xlabel(
        xlabel or f"Pressure ({pressure_unit})",
        fontsize=fontsize_label,
    )
    ax.set_ylabel(
        ylabel or f"Uptake ({uptake_unit})",
        fontsize=fontsize_label,
    )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if title:
        ax.set_title(title, fontsize=fontsize_label)

    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(fontsize=fontsize_legend)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)

    return str(Path(output).resolve())


def plot_mixture_isotherm(
    data_files: list[str],
    output: str = "mixture_isotherm_plot.png",
    dpi: int = 600,
    figsize: tuple[float, float] = (8, 6),
    adsorbates: list[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    labels: list[str] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    no_errorbars: bool = False,
    fontsize_label: int = 24,
    fontsize_tick: int = 16,
    fontsize_legend: int = 16,
) -> str:
    """Plot mixture isotherms (uptake vs relative humidity).

    Adsorbate names are auto-discovered from the JSON keys.
    Multiple files can be overlaid; when overlaying, legend
    entries are prefixed with the file label.

    Args:
        data_files: List of paths to mixture isotherm JSON files.
        output: Output image file path.
        dpi: Image resolution in dots per inch.
        figsize: Figure dimensions ``(width, height)`` in inches.
        adsorbates: Subset of adsorbate names to plot.  ``None``
            plots all discovered adsorbates.
        xlabel: Custom x-axis label.  Defaults to
            ``"Relative Humidity (%)"``.
        ylabel: Custom y-axis label.  Defaults to
            ``"Uptake (mol/kg)"``.
        title: Plot title.
        labels: Legend labels, one per data file.
        log_x: Use logarithmic x-axis.
        log_y: Use logarithmic y-axis.
        no_errorbars: If ``True``, omit error bars.
        fontsize_label: Font size for axis labels.
        fontsize_tick: Font size for tick labels.
        fontsize_legend: Font size for legend entries.

    Returns:
        The absolute path to the saved plot image.
    """
    plt = _get_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    multi_file = len(data_files) > 1

    marker_idx = 0
    for i, fpath in enumerate(data_files):
        parsed = load_isotherm(fpath)
        if parsed["format"] != "mixture_rh":
            raise ValueError(
                f"Expected mixture_rh format, got "
                f"'{parsed['format']}' in {fpath}"
            )

        file_label = (
            labels[i] if labels and i < len(labels) else Path(fpath).stem
        )

        ads_to_plot = adsorbates or parsed["adsorbates"]
        for ads in ads_to_plot:
            if ads not in parsed["uptakes"]:
                continue

            pretty = _pretty(ads)
            legend = f"{file_label} - {pretty}" if multi_file else pretty
            marker = _MARKERS[marker_idx % len(_MARKERS)]
            marker_idx += 1

            if no_errorbars:
                ax.plot(
                    parsed["rh_values"],
                    parsed["uptakes"][ads],
                    f"{marker}-",
                    label=legend,
                )
            else:
                ax.errorbar(
                    parsed["rh_values"],
                    parsed["uptakes"][ads],
                    yerr=parsed["errors"].get(ads),
                    fmt=f"{marker}-",
                    label=legend,
                    capsize=5,
                )

    ax.set_xlabel(
        xlabel or "Relative Humidity (%)",
        fontsize=fontsize_label,
    )
    ax.set_ylabel(
        ylabel or "Uptake (mol/kg)",
        fontsize=fontsize_label,
    )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if title:
        ax.set_title(title, fontsize=fontsize_label)

    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(fontsize=fontsize_legend)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)

    return str(Path(output).resolve())


def plot_selectivity(
    data_files: list[str],
    output: str = "selectivity_plot.png",
    dpi: int = 600,
    figsize: tuple[float, float] = (8, 6),
    selectivity_keys: list[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    labels: list[str] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    fontsize_label: int = 24,
    fontsize_tick: int = 16,
    fontsize_legend: int = 16,
) -> str:
    """Plot selectivity vs relative humidity from mixture data.

    Selectivity keys are auto-discovered from the JSON data
    (e.g. ``co2_n2_selectivity``).

    Args:
        data_files: List of paths to mixture isotherm JSON files.
        output: Output image file path.
        dpi: Image resolution in dots per inch.
        figsize: Figure dimensions ``(width, height)`` in inches.
        selectivity_keys: Specific selectivity fields to plot.
            ``None`` plots all discovered selectivity fields.
        xlabel: Custom x-axis label.
        ylabel: Custom y-axis label.
        title: Plot title.
        labels: Legend labels, one per data file.
        log_x: Use logarithmic x-axis.
        log_y: Use logarithmic y-axis.
        fontsize_label: Font size for axis labels.
        fontsize_tick: Font size for tick labels.
        fontsize_legend: Font size for legend entries.

    Returns:
        The absolute path to the saved plot image.
    """
    plt = _get_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    multi_file = len(data_files) > 1

    marker_idx = 0
    for i, fpath in enumerate(data_files):
        parsed = load_isotherm(fpath)
        if parsed["format"] != "mixture_rh":
            raise ValueError(
                f"Expected mixture_rh format, got "
                f"'{parsed['format']}' in {fpath}"
            )

        if not parsed["selectivity"]:
            raise ValueError(f"No selectivity data found in {fpath}")

        file_label = (
            labels[i] if labels and i < len(labels) else Path(fpath).stem
        )

        keys_to_plot = selectivity_keys or list(parsed["selectivity"].keys())
        for sel_key in keys_to_plot:
            if sel_key not in parsed["selectivity"]:
                continue

            # Make a readable legend label from the key
            # e.g. "co2_n2_selectivity" -> "CO2/N2"
            parts = sel_key.replace("_selectivity", "").split("_")
            pretty_sel = "/".join(_pretty(p) for p in parts)
            legend = (
                f"{file_label} - {pretty_sel}" if multi_file else pretty_sel
            )
            marker = _MARKERS[marker_idx % len(_MARKERS)]
            marker_idx += 1

            ax.plot(
                parsed["rh_values"],
                parsed["selectivity"][sel_key],
                f"{marker}-",
                label=legend,
            )

    ax.set_xlabel(
        xlabel or "Relative Humidity (%)",
        fontsize=fontsize_label,
    )
    ax.set_ylabel(
        ylabel or "Selectivity",
        fontsize=fontsize_label,
    )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if title:
        ax.set_title(title, fontsize=fontsize_label)

    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(fontsize=fontsize_legend)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)

    return str(Path(output).resolve())


def plot_isotherm(
    data_files: list[str],
    output: str | None = None,
    **kwargs,
) -> str:
    """Auto-detect format and plot isotherms.

    Inspects the first data file to determine the format, then
    dispatches to :func:`plot_single_isotherm` or
    :func:`plot_mixture_isotherm`.

    Args:
        data_files: List of paths to isotherm JSON files.
        output: Output image path.  Defaults depend on format.
        **kwargs: Forwarded to the format-specific plot function.

    Returns:
        The absolute path to the saved plot image.
    """
    parsed = load_isotherm(data_files[0])
    fmt = parsed["format"]

    if fmt == "single":
        return plot_single_isotherm(
            data_files,
            output=output or "isotherm_plot.png",
            **kwargs,
        )
    elif fmt == "mixture_rh":
        return plot_mixture_isotherm(
            data_files,
            output=output or "mixture_isotherm_plot.png",
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported format: {fmt}")
