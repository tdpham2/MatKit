"""Root conftest for pytest - exclude legacy test scripts."""

import os

collect_ignore = [
    os.path.join("tests", "run"),
    os.path.join("tests", "sample_raspa3"),
    os.path.join("tests", "remove_solvent.py"),
    os.path.join("tests", "test_parse_raspa2_pseudo_atom.py"),
]
