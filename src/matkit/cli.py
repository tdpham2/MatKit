import click
import json
from matkit.graspa import graspa
from matkit.raspa2 import raspa2
from matkit.tobacco import create_linker_from_smiles


from matkit.graspa_sycl import graspa_sycl


@click.group()
def main():
    """MatKit CLI: A modular toolkit for molecular simulations."""
    pass


# ==========================================
# GRASPA COMMANDS
# ==========================================
@main.group("graspa")
def graspa_cli():
    """Commands for gRASPA simulations."""
    pass


@graspa_cli.command("setup")
@click.option(
    "--cif",
    required=True,
    type=click.Path(exists=True),
    help="Path to input CIF file.",
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(),
    help="Directory to generate simulation files.",
)
@click.option(
    "--adsorbate",
    required=True,
    multiple=True,
    help="Adsorbate molecule name (e.g. CO2). Can be specified multiple times.",
)
@click.option("--temp", default=298.0, help="Temperature in Kelvin.")
@click.option("--pressure", default=1e5, help="Pressure in Pa.")
@click.option("--cutoff", default=12.8, help="Cutoff radius available.")
@click.option("--cycles", default=1000, help="Number of cycles.")
def graspa_setup(cif, outdir, adsorbate, temp, pressure, cutoff, cycles):
    """Setup gRASPA simulation files."""
    # Convert adsorbates tuple to list of dicts as expected by graspa.setup_simulation
    # The current graspa.setup_simulation expects a list of dicts with "MoleculeName"
    adsorbate_list = [{"MoleculeName": ads} for ads in adsorbate]

    try:
        graspa.setup_simulation(
            cif=cif,
            outpath=outdir,
            adsorbates=adsorbate_list,
            temperature=temp,
            pressure=pressure,
            cutoff=cutoff,
            n_cycle=cycles,
        )
        click.echo(f"Successfully set up gRASPA simulation in {outdir}")
    except Exception as e:
        click.echo(f"Error setting up gRASPA simulation: {e}", err=True)


@graspa_cli.command("analyze")
@click.option(
    "--path",
    required=True,
    type=click.Path(exists=True),
    help="Path to simulation output directory.",
)
@click.option(
    "--unit",
    default="mol/kg",
    type=click.Choice(["mol/kg", "mg/g", "g/L"]),
    help="Unit for uptake.",
)
def graspa_analyze(path, unit):
    """Analyze gRASPA simulation results."""
    try:
        result = graspa.get_output_data(path, unit=unit)
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error analyzing gRASPA results: {e}", err=True)


# ==========================================
# GRASPA SYCL COMMANDS
# ==========================================
@main.group("graspa_sycl")
def graspa_sycl_cli():
    """Commands for gRASPA SYCL simulations."""
    pass


@graspa_sycl_cli.command("setup")
@click.option(
    "--cif",
    required=True,
    type=click.Path(exists=True),
    help="Path to input CIF file.",
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(),
    help="Directory to generate simulation files.",
)
@click.option("--adsorbate", default="CO2", help="Adsorbate molecule name.")
@click.option("--temp", default=298.0, help="Temperature in Kelvin.")
@click.option("--pressure", default=1e5, help="Pressure in Pa.")
@click.option("--cutoff", default=12.8, help="Cutoff radius.")
@click.option("--cycles", default=1000, help="Number of cycles.")
def graspa_sycl_setup(cif, outdir, adsorbate, temp, pressure, cutoff, cycles):
    """Setup gRASPA SYCL simulation files."""
    try:
        graspa_sycl.setup_simulation(
            cif=cif,
            outpath=outdir,
            adsorbate=adsorbate,
            temperature=temp,
            pressure=pressure,
            cutoff=cutoff,
            n_cycle=cycles,
        )
        click.echo(f"Successfully set up gRASPA SYCL simulation in {outdir}")
    except Exception as e:
        click.echo(f"Error setting up gRASPA SYCL simulation: {e}", err=True)


@graspa_sycl_cli.command("analyze")
@click.option(
    "--path",
    required=True,
    type=click.Path(exists=True),
    help="Path to simulation output directory.",
)
@click.option(
    "--unit",
    default="mol/kg",
    type=click.Choice(["mol/kg", "g/L"]),
    help="Unit for uptake.",
)
@click.option(
    "--adsorbate",
    default="CO2",
    help="Adsorbate molecule name (for unit conversion).",
)
@click.option("--fname", default="raspa.log", help="Output filename.")
def graspa_sycl_analyze(path, unit, adsorbate, fname):
    """Analyze gRASPA SYCL simulation results."""
    try:
        result = graspa_sycl.get_output_data(
            output_path=path, unit=unit, adsorbate=adsorbate, output_fname=fname
        )
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error analyzing gRASPA SYCL results: {e}", err=True)


# ==========================================
# RASPA2 COMMANDS
# ==========================================
@main.group("raspa2")
def raspa2_cli():
    """Commands for RASPA2 simulations."""
    pass


@raspa2_cli.command("setup")
@click.option(
    "--cif",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Path to input CIF file(s).",
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(),
    help="Directory to generate simulation files.",
)
@click.option("--adsorbate", default="CO2", help="Adsorbate molecule name.")
@click.option("--temp", default=298.0, help="Temperature in Kelvin.")
@click.option("--pressure", default=1e5, help="Pressure in Pa.")
@click.option("--cutoff", default=12.8, help="Cutoff radius.")
@click.option("--cycles", default=1000, help="Number of cycles.")
def raspa2_setup(cif, outdir, adsorbate, temp, pressure, cutoff, cycles):
    """Setup RASPA2 simulation files."""
    try:
        # raspa2.setup_input_simulation takes a list of cifs
        raspa2.setup_input_simulation(
            cifs=list(cif),
            outpath=outdir,
            adsorbate=adsorbate,
            temperature=temp,
            pressure=pressure,
            cutoff=cutoff,
            n_cycle=cycles,
        )
        click.echo(f"Successfully set up RASPA2 simulation in {outdir}")
    except Exception as e:
        click.echo(f"Error setting up RASPA2 simulation: {e}", err=True)


@raspa2_cli.command("analyze")
@click.option(
    "--path",
    required=True,
    type=click.Path(exists=True),
    help="Path to simulation output file (raspa.log or similar).",
)
@click.option(
    "--unit",
    default="mol/kg",
    type=click.Choice(["mol/kg", "g/L"]),
    help="Unit for uptake.",
)
def raspa2_analyze(path, unit):
    """Analyze RASPA2 simulation results."""
    try:
        result = raspa2.get_output_data(path, unit=unit)
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error analyzing RASPA2 results: {e}", err=True)


# ==========================================
# TOBACCO COMMANDS
# ==========================================
@main.group("tobacco")
def tobacco_cli():
    """Commands for Tobacco linker generation."""
    pass


@tobacco_cli.command("create")
@click.option("--smiles", required=True, help="SMILES string for the linker.")
@click.option(
    "--site",
    required=True,
    multiple=True,
    help="Connection site atom label (e.g. N1).",
)
@click.option("--out", default="final_output.cif", help="Output CIF filename.")
def tobacco_create(smiles, site, out):
    """Create a linker CIF from SMILES."""
    try:
        create_linker_from_smiles.create_linker(
            smiles=smiles, connection_sites=list(site), output_cif=out
        )
    except Exception as e:
        click.echo(f"Error creating linker: {e}", err=True)


# ==========================================
# PLOT COMMANDS
# ==========================================
@main.group("plot")
def plot_cli():
    """Commands for plotting simulation data."""
    pass


@plot_cli.command("isotherm")
@click.option(
    "--data",
    multiple=True,
    type=click.Path(exists=True),
    help="Path to isotherm JSON file. Can be specified multiple "
    "times to overlay plots.",
)
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing isotherm JSON files. "
    "All *.json files are loaded and overlaid.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output image file path (default: isotherm_plot.png "
    "or mixture_isotherm_plot.png).",
)
@click.option("--dpi", default=600, help="Image resolution in DPI.")
@click.option(
    "--figsize",
    nargs=2,
    type=float,
    default=(8, 6),
    help="Figure size as WIDTH HEIGHT in inches.",
)
@click.option(
    "--adsorbate",
    multiple=True,
    help="Adsorbate(s) to include in mixture plots. "
    "Can be specified multiple times. "
    "Omit to plot all discovered adsorbates.",
)
@click.option(
    "--label",
    multiple=True,
    help="Legend label for each --data file. Can be specified multiple times.",
)
@click.option("--xlabel", default=None, help="Custom x-axis label.")
@click.option("--ylabel", default=None, help="Custom y-axis label.")
@click.option("--title", default=None, help="Plot title.")
@click.option("--log-x", is_flag=True, help="Use logarithmic x-axis.")
@click.option("--log-y", is_flag=True, help="Use logarithmic y-axis.")
@click.option(
    "--no-errorbars",
    is_flag=True,
    help="Omit error bars from the plot.",
)
@click.option(
    "--fontsize-label",
    default=24,
    help="Font size for axis labels.",
)
@click.option(
    "--fontsize-tick",
    default=16,
    help="Font size for tick labels.",
)
@click.option(
    "--fontsize-legend",
    default=16,
    help="Font size for legend text.",
)
def plot_isotherm_cmd(
    data,
    data_dir,
    output,
    dpi,
    figsize,
    adsorbate,
    label,
    xlabel,
    ylabel,
    title,
    log_x,
    log_y,
    no_errorbars,
    fontsize_label,
    fontsize_tick,
    fontsize_legend,
):
    """Plot isotherms from simulation JSON data.

    Auto-detects the data format (single-component pressure
    isotherm or mixture RH isotherm) and generates the
    appropriate plot. Multiple --data files or a --data-dir
    can be specified to overlay isotherms for comparison.

    \b
    Examples:
      matkit plot isotherm --data CO2_isotherm_298K.json
      matkit plot isotherm --data-dir results/
      matkit plot isotherm --data r1.json --data r2.json
      matkit plot isotherm --data mixture.json --adsorbate co2
      matkit plot isotherm --data CO2.json --log-x
    """
    try:
        from matkit.plot.parsers import (
            collect_data_files,
            load_isotherm,
        )

        files = collect_data_files(data=data, data_dir=data_dir)

        # Detect format from first file to route correctly
        parsed = load_isotherm(files[0])
        fmt = parsed["format"]

        common_kwargs = dict(
            output=output,
            dpi=dpi,
            figsize=figsize,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            labels=list(label) if label else None,
            log_x=log_x,
            log_y=log_y,
            no_errorbars=no_errorbars,
            fontsize_label=fontsize_label,
            fontsize_tick=fontsize_tick,
            fontsize_legend=fontsize_legend,
        )

        if fmt == "mixture_rh":
            from matkit.plot.isotherm import (
                plot_mixture_isotherm,
            )

            common_kwargs["adsorbates"] = list(adsorbate) if adsorbate else None
            if output is None:
                common_kwargs["output"] = "mixture_isotherm_plot.png"
            out = plot_mixture_isotherm(files, **common_kwargs)
        else:
            from matkit.plot.isotherm import (
                plot_single_isotherm,
            )

            if output is None:
                common_kwargs["output"] = "isotherm_plot.png"
            out = plot_single_isotherm(files, **common_kwargs)

        click.echo(f"Plot saved to {out}")

    except ImportError:
        click.echo(
            "Error: matplotlib is required for plotting. "
            "Install with: pip install matkit[plot]",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@plot_cli.command("selectivity")
@click.option(
    "--data",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Path to mixture isotherm JSON file. Can be specified multiple times.",
)
@click.option(
    "--output",
    default="selectivity_plot.png",
    type=click.Path(),
    help="Output image file path.",
)
@click.option("--dpi", default=600, help="Image resolution in DPI.")
@click.option(
    "--figsize",
    nargs=2,
    type=float,
    default=(8, 6),
    help="Figure size as WIDTH HEIGHT in inches.",
)
@click.option(
    "--selectivity-key",
    multiple=True,
    help="Selectivity field(s) to plot (e.g. co2_n2_selectivity). "
    "Omit to plot all discovered selectivity fields.",
)
@click.option(
    "--label",
    multiple=True,
    help="Legend label for each --data file.",
)
@click.option("--xlabel", default=None, help="Custom x-axis label.")
@click.option("--ylabel", default=None, help="Custom y-axis label.")
@click.option("--title", default=None, help="Plot title.")
@click.option("--log-x", is_flag=True, help="Use logarithmic x-axis.")
@click.option("--log-y", is_flag=True, help="Use logarithmic y-axis.")
@click.option(
    "--fontsize-label",
    default=24,
    help="Font size for axis labels.",
)
@click.option(
    "--fontsize-tick",
    default=16,
    help="Font size for tick labels.",
)
@click.option(
    "--fontsize-legend",
    default=16,
    help="Font size for legend text.",
)
def plot_selectivity_cmd(
    data,
    output,
    dpi,
    figsize,
    selectivity_key,
    label,
    xlabel,
    ylabel,
    title,
    log_x,
    log_y,
    fontsize_label,
    fontsize_tick,
    fontsize_legend,
):
    """Plot selectivity vs relative humidity from mixture data.

    Selectivity keys are auto-discovered from the JSON data
    (e.g. co2_n2_selectivity). Multiple --data files can be
    overlaid for comparison.

    \b
    Examples:
      matkit plot selectivity --data mixture.json
      matkit plot selectivity --data m1.json --data m2.json
      matkit plot selectivity --data mixture.json --log-y
    """
    try:
        from matkit.plot.isotherm import plot_selectivity

        out = plot_selectivity(
            data_files=list(data),
            output=output,
            dpi=dpi,
            figsize=figsize,
            selectivity_keys=(
                list(selectivity_key) if selectivity_key else None
            ),
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            labels=list(label) if label else None,
            log_x=log_x,
            log_y=log_y,
            fontsize_label=fontsize_label,
            fontsize_tick=fontsize_tick,
            fontsize_legend=fontsize_legend,
        )
        click.echo(f"Plot saved to {out}")

    except ImportError:
        click.echo(
            "Error: matplotlib is required for plotting. "
            "Install with: pip install matkit[plot]",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    main()
