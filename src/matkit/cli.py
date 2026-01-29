import click
import os
import json
from pathlib import Path
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


if __name__ == "__main__":
    main()
