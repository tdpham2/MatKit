import logging

import click
import json


@click.group()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (-v for info, -vv for debug).",
)
def main(verbose):
    """MatKit CLI: A modular toolkit for molecular simulations."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
    )


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
    from matkit.graspa import setup_simulation

    # Convert adsorbates tuple to list of dicts
    adsorbate_list = [{"MoleculeName": ads} for ads in adsorbate]

    try:
        setup_simulation(
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


@graspa_cli.command("batch-setup")
@click.option(
    "--cif-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing CIF files.",
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(),
    help="Base output directory for simulation files.",
)
@click.option(
    "--adsorbate",
    required=True,
    multiple=True,
    help="Adsorbate molecule name (e.g. CO2). Can be specified multiple times.",
)
@click.option(
    "--temp",
    required=True,
    multiple=True,
    type=float,
    help="Temperature in Kelvin. Can be specified multiple times.",
)
@click.option(
    "--pressure",
    required=True,
    multiple=True,
    type=float,
    help="Pressure in Pa. Can be specified multiple times.",
)
@click.option("--cutoff", default=12.8, help="Cutoff radius in Angstrom.")
@click.option("--cycles", default=1000, help="Number of MC cycles.")
@click.option(
    "--workers",
    default=None,
    type=int,
    help="Max parallel threads for CIF processing.",
)
def graspa_batch_setup(
    cif_dir, outdir, adsorbate, temp, pressure, cutoff, cycles, workers
):
    """Set up gRASPA simulations for all CIF x T x P."""
    from matkit.graspa import setup_batch

    adsorbate_list = [{"MoleculeName": ads} for ads in adsorbate]

    try:
        manifest = setup_batch(
            cif_dir=cif_dir,
            outpath=outdir,
            adsorbates=adsorbate_list,
            temperatures=list(temp),
            pressures=list(pressure),
            cutoff=cutoff,
            n_cycle=cycles,
            max_workers=workers,
        )
        click.echo(f"Set up {len(manifest)} simulations in {outdir}")
        click.echo(f"Manifest written to {outdir}/simulations.jsonl")
    except Exception as e:
        click.echo(f"Error setting up batch simulations: {e}", err=True)


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
    from matkit.graspa import get_output_data

    try:
        result = get_output_data(path, unit=unit)
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
    from matkit.graspa_sycl import setup_simulation

    try:
        setup_simulation(
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
    from matkit.graspa_sycl import get_output_data

    try:
        result = get_output_data(
            output_path=path, unit=unit, adsorbate=adsorbate, output_fname=fname
        )
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error analyzing gRASPA SYCL results: {e}", err=True)


# ==========================================
# PYGRASPA COMMANDS (ML-potential GCMC)
# ==========================================
@main.group("pygraspa")
def pygraspa_cli():
    """Commands for pygRASPA simulations (ML-potential GCMC).

    Writes simulation files plus a ``run.py`` launcher. Execute on a GPU
    machine with pygRASPA + the required ML backend installed.
    """
    pass


def _parse_ecomps(ctx, param, value):
    """Click callback: convert comma-separated floats into a list."""
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise click.BadParameter(
            f"--ecomps must be comma-separated floats: {e}"
        )


@pygraspa_cli.command("setup")
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
    help="Adsorbate molecule name (e.g. CO2). Can be repeated.",
)
@click.option(
    "--model",
    "model_path",
    required=True,
    help="ML model checkpoint path.",
)
@click.option(
    "--model-type",
    default="FAIRChem-esen",
    type=click.Choice(
        [
            "FAIRChem-esen",
            "FAIRChem-uma",
            "FAIRChem-AllScAIP",
            "mace_polar",
            "Allegro",
            "customized",
        ]
    ),
    help="ML backend identifier.",
)
@click.option(
    "--task",
    default=None,
    help="Task name (FAIRChem-uma / AllScAIP).",
)
@click.option(
    "--ecomps",
    required=True,
    callback=_parse_ecomps,
    help="Comma-separated E_comp values (eV), one per --adsorbate.",
)
@click.option(
    "--mode",
    default="run-auto",
    type=click.Choice(["run", "run-auto"]),
    help="pygRASPA execution mode written into run.py.",
)
@click.option("--save-poscar", is_flag=True, help="Save accepted POSCARs.")
@click.option("--temp", default=298.0, help="Temperature in Kelvin.")
@click.option("--pressure", default=1e5, help="Pressure in Pa.")
@click.option("--cutoff", default=12.8, help="Cutoff radius (Angstrom).")
@click.option("--cycles", default=1000, help="Number of MC cycles.")
@click.option(
    "--template-dir",
    default="template",
    help=(
        "Template subdir (template, template_mixture, "
        "template_mixture_isotherm)."
    ),
)
def pygraspa_setup(
    cif,
    outdir,
    adsorbate,
    model_path,
    model_type,
    task,
    ecomps,
    mode,
    save_poscar,
    temp,
    pressure,
    cutoff,
    cycles,
    template_dir,
):
    """Setup a pygRASPA simulation."""
    from matkit.pygraspa import setup_simulation

    adsorbate_list = [{"MoleculeName": ads} for ads in adsorbate]
    try:
        setup_simulation(
            cif=cif,
            outpath=outdir,
            adsorbates=adsorbate_list,
            model_path=model_path,
            model_type=model_type,
            E_comps=ecomps,
            task=task,
            mode=mode,
            save_poscar=save_poscar,
            temperature=temp,
            pressure=pressure,
            cutoff=cutoff,
            n_cycle=cycles,
            template_dir=template_dir,
        )
        click.echo(f"Successfully set up pygRASPA simulation in {outdir}")
    except Exception as e:
        click.echo(f"Error setting up pygRASPA simulation: {e}", err=True)


@pygraspa_cli.command("batch-setup")
@click.option(
    "--cif-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing CIF files.",
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(),
    help="Base output directory.",
)
@click.option(
    "--adsorbate",
    required=True,
    multiple=True,
    help="Adsorbate molecule name. Can be repeated.",
)
@click.option(
    "--model",
    "model_path",
    required=True,
    help="ML model checkpoint path.",
)
@click.option(
    "--model-type",
    default="FAIRChem-esen",
    type=click.Choice(
        [
            "FAIRChem-esen",
            "FAIRChem-uma",
            "FAIRChem-AllScAIP",
            "mace_polar",
            "Allegro",
            "customized",
        ]
    ),
    help="ML backend identifier.",
)
@click.option(
    "--task",
    default=None,
    help="Task name (FAIRChem-uma / AllScAIP).",
)
@click.option(
    "--ecomps",
    required=True,
    callback=_parse_ecomps,
    help="Comma-separated E_comp values, one per --adsorbate.",
)
@click.option(
    "--mode",
    default="run-auto",
    type=click.Choice(["run", "run-auto"]),
    help="pygRASPA execution mode written into run.py.",
)
@click.option("--save-poscar", is_flag=True, help="Save accepted POSCARs.")
@click.option(
    "--temp",
    required=True,
    multiple=True,
    type=float,
    help="Temperature(s) in Kelvin. Can be repeated.",
)
@click.option(
    "--pressure",
    required=True,
    multiple=True,
    type=float,
    help="Pressure(s) in Pa. Can be repeated.",
)
@click.option("--cutoff", default=12.8, help="Cutoff radius (Angstrom).")
@click.option("--cycles", default=1000, help="Number of MC cycles.")
@click.option(
    "--template-dir",
    default="template",
    help="Template subdir.",
)
@click.option(
    "--workers",
    default=None,
    type=int,
    help="Max parallel threads for CIF processing.",
)
def pygraspa_batch_setup(
    cif_dir,
    outdir,
    adsorbate,
    model_path,
    model_type,
    task,
    ecomps,
    mode,
    save_poscar,
    temp,
    pressure,
    cutoff,
    cycles,
    template_dir,
    workers,
):
    """Set up pygRASPA simulations for all CIF x T x P."""
    from matkit.pygraspa import setup_batch

    adsorbate_list = [{"MoleculeName": ads} for ads in adsorbate]
    try:
        manifest = setup_batch(
            cif_dir=cif_dir,
            outpath=outdir,
            adsorbates=adsorbate_list,
            model_path=model_path,
            model_type=model_type,
            E_comps=ecomps,
            task=task,
            mode=mode,
            save_poscar=save_poscar,
            temperatures=list(temp),
            pressures=list(pressure),
            cutoff=cutoff,
            n_cycle=cycles,
            template_dir=template_dir,
            max_workers=workers,
        )
        click.echo(f"Set up {len(manifest)} simulations in {outdir}")
        click.echo(f"Manifest written to {outdir}/simulations.jsonl")
    except Exception as e:
        click.echo(f"Error setting up batch simulations: {e}", err=True)


@pygraspa_cli.command("compute-ecomp")
@click.option(
    "--def-file",
    "def_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to adsorbate .def file.",
)
@click.option(
    "--model",
    "model_path",
    required=True,
    help="ML model checkpoint path.",
)
@click.option(
    "--model-type",
    default="FAIRChem-esen",
    type=click.Choice(
        [
            "FAIRChem-esen",
            "FAIRChem-uma",
            "FAIRChem-AllScAIP",
            "mace_polar",
            "Allegro",
            "customized",
        ]
    ),
)
@click.option(
    "--task",
    default=None,
    help="Task name (FAIRChem-uma / AllScAIP).",
)
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]))
@click.option(
    "--cache",
    "cache_path",
    default=None,
    type=click.Path(),
    help="JSON cache file to memoize results.",
)
def pygraspa_compute_ecomp(
    def_file,
    model_path,
    model_type,
    task,
    device,
    cache_path,
):
    """Compute E_comp (isolated-adsorbate ML energy) via pygRASPA."""
    from matkit.pygraspa import compute_ecomp

    try:
        value = compute_ecomp(
            adsorbate_def=def_file,
            model_path=model_path,
            model_type=model_type,
            task=task,
            device=device,
            cache_path=cache_path,
        )
        click.echo(
            json.dumps(
                {"adsorbate_def": def_file, "E_comp_eV": value},
                indent=2,
            )
        )
    except Exception as e:
        click.echo(f"Error computing E_comp: {e}", err=True)


@pygraspa_cli.command("analyze")
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
@click.option("--fname", default="output.log", help="Log filename.")
@click.option(
    "--skip-cycles",
    default=None,
    type=int,
    help="Override n_skip_cycles (default: from simulation.input).",
)
def pygraspa_analyze(path, unit, fname, skip_cycles):
    """Analyze pygRASPA simulation results."""
    from matkit.pygraspa import get_output_data

    try:
        result = get_output_data(
            output_path=path,
            unit=unit,
            output_fname=fname,
            n_skip_cycles=skip_cycles,
        )
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error analyzing pygRASPA results: {e}", err=True)


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
    from matkit.raspa2 import setup_input_simulation

    try:
        setup_input_simulation(
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
    from matkit.raspa2 import get_output_data

    try:
        result = get_output_data(path, unit=unit)
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
    from matkit.tobacco import create_linker

    try:
        create_linker(
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


# ==========================================
# MLIP COMMANDS
# ==========================================
@main.group("mlip")
def mlip_cli():
    """Commands for ML interatomic potential calculations."""
    pass


@mlip_cli.command("mace-opt")
@click.option(
    "--fname",
    required=True,
    type=click.Path(exists=True),
    help="Input structure file (CIF, XYZ, POSCAR, etc.).",
)
@click.option(
    "--run-type",
    default="geo_opt",
    type=click.Choice(["geo_opt", "cell_opt", "geo_opt_cell_opt"]),
    help="Optimization type.",
)
@click.option("--steps", default=1000, help="Max optimization steps.")
@click.option(
    "--fmax",
    default=1e-3,
    help="Force convergence criterion (eV/A).",
)
@click.option(
    "--model",
    default="medium",
    type=click.Choice(["small", "medium", "large"]),
    help="MACE-MP model size.",
)
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(["cpu", "cuda"]),
    help="Compute device.",
)
@click.option(
    "--dispersion/--no-dispersion",
    default=True,
    help="Include D3 dispersion corrections.",
)
@click.option(
    "--default-dtype",
    default="float64",
    type=click.Choice(["float32", "float64"]),
    help="Floating point precision.",
)
@click.option(
    "--write-traj",
    is_flag=True,
    help="Write ASE trajectory files.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output filename (auto-generated if omitted).",
)
def mace_opt_cmd(
    fname,
    run_type,
    steps,
    fmax,
    model,
    device,
    dispersion,
    default_dtype,
    write_traj,
    output,
):
    """Run MACE-MP geometry/cell optimization."""
    try:
        from matkit.mlip import run_opt_mace

        run_opt_mace(
            fname=fname,
            run_type=run_type,
            steps=steps,
            fmax=fmax,
            model=model,
            device=device,
            dispersion=dispersion,
            default_dtype=default_dtype,
            write_traj=write_traj,
            output_fname=output,
        )
        click.echo("MACE-MP optimization complete.")
    except ImportError:
        click.echo(
            "Error: mace-torch required. "
            "Install with: pip install matkit[mlip]",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@mlip_cli.command("uma-opt")
@click.option(
    "--fname",
    required=True,
    type=click.Path(exists=True),
    help="Input structure file.",
)
@click.option(
    "--run-type",
    default="geo_opt",
    type=click.Choice(["geo_opt", "cell_opt", "geo_opt_cell_opt"]),
    help="Optimization type.",
)
@click.option("--steps", default=1000, help="Max optimization steps.")
@click.option(
    "--fmax",
    default=1e-3,
    help="Force convergence criterion (eV/A).",
)
@click.option(
    "--model",
    default="uma-s-1p2",
    help="UMA model name (e.g. uma-s-1p2, uma-m-1p1).",
)
@click.option(
    "--task-name",
    default="omat",
    type=click.Choice(["omat", "oc20", "oc22", "oc25", "omol", "odac", "omc"]),
    help="Task head for domain-specific prediction.",
)
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(["cpu", "cuda"]),
    help="Compute device.",
)
@click.option(
    "--write-traj",
    is_flag=True,
    help="Write ASE trajectory files.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output filename (auto-generated if omitted).",
)
def uma_opt_cmd(
    fname,
    run_type,
    steps,
    fmax,
    model,
    task_name,
    device,
    write_traj,
    output,
):
    """Run UMA geometry/cell optimization."""
    try:
        from matkit.mlip.uma import run_opt_uma

        run_opt_uma(
            fname=fname,
            run_type=run_type,
            steps=steps,
            fmax=fmax,
            model=model,
            task_name=task_name,
            device=device,
            write_traj=write_traj,
            output_fname=output,
        )
        click.echo("UMA optimization complete.")
    except ImportError:
        click.echo(
            "Error: fairchem-core required. "
            "Install with: pip install matkit[uma]",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@mlip_cli.command("uma-sp")
@click.option(
    "--fname",
    required=True,
    type=click.Path(exists=True),
    help="Input structure file.",
)
@click.option(
    "--model",
    default="uma-s-1p2",
    help="UMA model name.",
)
@click.option(
    "--task-name",
    default="omat",
    type=click.Choice(["omat", "oc20", "oc22", "oc25", "omol", "odac", "omc"]),
    help="Task head for domain-specific prediction.",
)
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(["cpu", "cuda"]),
    help="Compute device.",
)
def uma_sp_cmd(fname, model, task_name, device):
    """Run UMA single-point energy calculation."""
    try:
        from matkit.mlip.uma import run_sp_uma

        result = run_sp_uma(
            fname=fname,
            model=model,
            task_name=task_name,
            device=device,
        )
        click.echo(json.dumps(result, indent=2))
    except ImportError:
        click.echo(
            "Error: fairchem-core required. "
            "Install with: pip install matkit[uma]",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@mlip_cli.command("uma-md")
@click.option(
    "--fname",
    required=True,
    type=click.Path(exists=True),
    help="Input structure file.",
)
@click.option(
    "--model",
    default="uma-s-1p2",
    help="UMA model name.",
)
@click.option(
    "--task-name",
    default="omat",
    type=click.Choice(["omat", "oc20", "oc22", "oc25", "omol", "odac", "omc"]),
    help="Task head for domain-specific prediction.",
)
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(["cpu", "cuda"]),
    help="Compute device.",
)
@click.option(
    "--temperature",
    default=300.0,
    help="Target temperature in Kelvin.",
)
@click.option(
    "--timestep",
    default=1.0,
    help="MD timestep in femtoseconds.",
)
@click.option("--steps", default=1000, help="Number of MD steps.")
@click.option(
    "--ensemble",
    default="nvt",
    type=click.Choice(["nve", "nvt"]),
    help="MD ensemble.",
)
@click.option(
    "--friction",
    default=0.01,
    help="Langevin friction coefficient (NVT only).",
)
@click.option(
    "--write-traj",
    is_flag=True,
    help="Write ASE trajectory file.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output filename (auto-generated if omitted).",
)
@click.option(
    "--log-interval",
    default=10,
    help="Steps between log entries.",
)
def uma_md_cmd(
    fname,
    model,
    task_name,
    device,
    temperature,
    timestep,
    steps,
    ensemble,
    friction,
    write_traj,
    output,
    log_interval,
):
    """Run UMA molecular dynamics."""
    try:
        from matkit.mlip.uma import run_md_uma

        run_md_uma(
            fname=fname,
            model=model,
            task_name=task_name,
            device=device,
            temperature=temperature,
            timestep=timestep,
            steps=steps,
            ensemble=ensemble,
            friction=friction,
            write_traj=write_traj,
            output_fname=output,
            log_interval=log_interval,
        )
        click.echo("UMA MD complete.")
    except ImportError:
        click.echo(
            "Error: fairchem-core required. "
            "Install with: pip install matkit[uma]",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@mlip_cli.command("uma-opt-batch")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input CIF file or directory of CIF files.",
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(),
    help="Output directory for CIFs and results.jsonl.",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    default=["uma-s-1p2"],
    help="UMA model name(s). Can be specified multiple times.",
)
@click.option(
    "--run-type",
    "run_types",
    multiple=True,
    default=["geo_opt"],
    type=click.Choice(["geo_opt", "cell_opt", "geo_opt_cell_opt"]),
    help="Optimization type(s). Can be specified multiple times.",
)
@click.option(
    "--task-name",
    default="omat",
    type=click.Choice(["omat", "oc20", "oc22", "oc25", "omol", "odac", "omc"]),
    help="Task head for domain-specific prediction.",
)
@click.option("--steps", default=1000, help="Max optimization steps.")
@click.option(
    "--fmax",
    default=1e-3,
    help="Force convergence criterion (eV/A) for geometry.",
)
@click.option(
    "--fmax-cell",
    default=None,
    type=float,
    help="Force convergence for cell opt. Defaults to --fmax.",
)
@click.option(
    "--num-gpus",
    default=None,
    type=int,
    help="Number of GPUs. Auto-detected if omitted.",
)
@click.option(
    "--device",
    default="cuda",
    type=click.Choice(["cpu", "cuda"]),
    help="Compute device.",
)
@click.option(
    "--write-traj",
    is_flag=True,
    help="Write ASE trajectory files.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing output files.",
)
def uma_opt_batch_cmd(
    input_path,
    outdir,
    models,
    run_types,
    task_name,
    steps,
    fmax,
    fmax_cell,
    num_gpus,
    device,
    write_traj,
    overwrite,
):
    """Run UMA optimization in batch across multiple GPUs."""
    try:
        from matkit.mlip.uma import run_opt_uma_batch

        result_path = run_opt_uma_batch(
            input_path=input_path,
            output_dir=outdir,
            models=list(models),
            run_types=list(run_types),
            task_name=task_name,
            steps=steps,
            fmax=fmax,
            fmax_cell=fmax_cell,
            num_gpus=num_gpus,
            device=device,
            write_traj=write_traj,
            overwrite=overwrite,
        )
        click.echo(f"Results written to: {result_path}")
    except ImportError:
        click.echo(
            "Error: fairchem-core required. "
            "Install with: pip install matkit[uma]",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


# ==========================================
# PACMOF2 COMMANDS
# ==========================================
@main.group("pacmof2")
def pacmof2_cli():
    """Commands for PACMOF2 charge prediction."""
    pass


@pacmof2_cli.command("predict")
@click.option(
    "--cif",
    default=None,
    type=click.Path(exists=True),
    help="Path to a single CIF file.",
)
@click.option(
    "--cif-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing CIF files.",
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(),
    help="Output directory for CIFs with predicted charges.",
)
@click.option(
    "--identifier",
    default="_pacmof",
    help="Suffix for output filenames (default: _pacmof).",
)
@click.option(
    "--net-charge",
    default="0",
    help="Net charge: integer for single MOF, or path to "
    "JSON file mapping filenames to charges for batch.",
)
@click.option(
    "--adjust-method",
    default="mean",
    type=click.Choice(["mean", "magnitude"]),
    help="Charge adjustment method.",
)
def pacmof2_predict(
    cif,
    cif_dir,
    outdir,
    identifier,
    net_charge,
    adjust_method,
):
    """Predict partial atomic charges for CIF structures.

    Provide either --cif for a single file or --cif-dir for
    a directory of CIF files.

    \b
    Examples:
      matkit pacmof2 predict --cif-dir cifs/ --outdir charged/
      matkit pacmof2 predict --cif structure.cif --outdir charged/
      matkit pacmof2 predict --cif-dir cifs/ --outdir charged/ \\
          --net-charge charges.json
    """
    if cif and cif_dir:
        click.echo(
            "Error: specify --cif or --cif-dir, not both.",
            err=True,
        )
        return
    if not cif and not cif_dir:
        click.echo("Error: provide --cif or --cif-dir.", err=True)
        return

    cif_path = cif_dir if cif_dir else cif

    # Parse net_charge: try int/float first, then JSON path
    try:
        nc = int(net_charge)
    except ValueError:
        try:
            nc = float(net_charge)
        except ValueError:
            nc = net_charge  # treat as JSON file path

    try:
        from matkit.pacmof2 import run_charge_prediction

        result = run_charge_prediction(
            cif_path=cif_path,
            output_dir=outdir,
            identifier=identifier,
            net_charge=nc,
            adjust_charge_method=adjust_method,
        )
        click.echo(
            f"Predicted charges for {result['num_structures']} "
            f"structure(s). Output: {result['output_dir']}"
        )
    except ImportError:
        click.echo(
            "Error: pacmof2 is required. "
            "Install with: pip install matkit[pacmof2]",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


# ==========================================
# ZEOPP COMMANDS
# ==========================================
@main.group("zeopp")
def zeopp_cli():
    """Commands for Zeo++ pore analysis."""
    pass


@zeopp_cli.command("run")
@click.option(
    "--cif",
    required=True,
    type=click.Path(exists=True),
    help="Path to input CIF file.",
)
@click.option(
    "--analysis",
    multiple=True,
    default=["res"],
    type=click.Choice(["res", "sa", "vol", "psd", "chan"]),
    help="Analysis type. Can be specified multiple times.",
)
@click.option(
    "--probe-radius",
    default=1.86,
    help="Probe radius in Angstrom.",
)
@click.option(
    "--chan-radius",
    default=1.86,
    help="Channel radius in Angstrom.",
)
@click.option(
    "--num-samples",
    default=2000,
    help="Number of Monte Carlo samples.",
)
@click.option(
    "--ha/--no-ha",
    default=True,
    help="Use high accuracy mode (-ha). Default: enabled.",
)
@click.option(
    "--radii",
    default=None,
    type=click.Path(exists=True),
    help="Path to atomic radii file. Default: bundled UFF.rad.",
)
@click.option(
    "--network-path",
    default=None,
    type=click.Path(),
    help="Path to Zeo++ network binary.",
)
@click.option(
    "--outdir",
    default=None,
    type=click.Path(),
    help="Output directory for result files.",
)
def zeopp_run(
    cif,
    analysis,
    probe_radius,
    chan_radius,
    num_samples,
    ha,
    radii,
    network_path,
    outdir,
):
    """Run Zeo++ analysis on a CIF structure."""
    from matkit.zeopp import run_zeopp

    try:
        result = run_zeopp(
            cif=cif,
            analyses=list(analysis),
            probe_radius=probe_radius,
            chan_radius=chan_radius,
            num_samples=num_samples,
            ha=ha,
            radii_file=radii,
            network_path=network_path,
            output_dir=outdir,
        )
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error running Zeo++: {e}", err=True)


@zeopp_cli.command("analyze")
@click.option(
    "--path",
    required=True,
    type=click.Path(exists=True),
    help="Path to Zeo++ output directory or file.",
)
@click.option(
    "--analysis",
    multiple=True,
    type=click.Choice(["res", "sa", "vol", "psd", "chan"]),
    help="Analysis types to parse. Auto-detects if omitted.",
)
def zeopp_analyze(path, analysis):
    """Parse existing Zeo++ output files."""
    from matkit.zeopp import get_output_data

    try:
        analyses = list(analysis) if analysis else None
        result = get_output_data(path, analyses=analyses)
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error analyzing Zeo++ results: {e}", err=True)


@zeopp_cli.command("run-batch")
@click.option(
    "--cif-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing CIF files.",
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(),
    help="Output directory for results.",
)
@click.option(
    "--analysis",
    multiple=True,
    default=["res"],
    type=click.Choice(["res", "sa", "vol", "psd", "chan"]),
    help="Analysis type. Can be specified multiple times.",
)
@click.option(
    "--probe-radius",
    default=1.86,
    help="Probe radius in Angstrom.",
)
@click.option(
    "--chan-radius",
    default=1.86,
    help="Channel radius in Angstrom.",
)
@click.option(
    "--num-samples",
    default=2000,
    help="Number of Monte Carlo samples.",
)
@click.option(
    "--ha/--no-ha",
    default=True,
    help="Use high accuracy mode (-ha). Default: enabled.",
)
@click.option(
    "--radii",
    default=None,
    type=click.Path(exists=True),
    help="Path to atomic radii file. Default: bundled UFF.rad.",
)
@click.option(
    "--network-path",
    default=None,
    type=click.Path(),
    help="Path to Zeo++ network binary.",
)
@click.option(
    "--max-workers",
    default=None,
    type=int,
    help="Max parallel processes. Default: CPU count.",
)
def zeopp_run_batch(
    cif_dir,
    outdir,
    analysis,
    probe_radius,
    chan_radius,
    num_samples,
    ha,
    radii,
    network_path,
    max_workers,
):
    """Run Zeo++ on all CIFs in a directory.

    Processes CIF files in parallel and writes results
    to a consolidated results.jsonl file.

    \b
    Examples:
      matkit zeopp run-batch --cif-dir cifs/ --outdir out/
      matkit zeopp run-batch --cif-dir cifs/ --outdir out/ \\
          --analysis res --analysis sa --max-workers 32
    """
    from matkit.zeopp import run_batch

    try:
        summary = run_batch(
            cif_dir=cif_dir,
            output_dir=outdir,
            analyses=list(analysis),
            probe_radius=probe_radius,
            chan_radius=chan_radius,
            num_samples=num_samples,
            ha=ha,
            radii_file=radii,
            network_path=network_path,
            max_workers=max_workers,
        )
        click.echo(f"Results written to {summary}")
    except Exception as e:
        click.echo(f"Error in batch run: {e}", err=True)


if __name__ == "__main__":
    main()
