from ase.calculators.orca import OrcaProfile
from ase.calculators.orca import ORCA


def run_orca(
    orca_command: str,
    run_type: str,
):
    profile = OrcaProfile(command=orca_command)
    _calc = ORCA(profile=profile)
