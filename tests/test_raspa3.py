"""Tests for matkit.raspa3 module (RASPA2 -> RASPA3 format conversion)."""

import json
import pytest
from pathlib import Path

from matkit.raspa3.raspa3 import (
    parse_raspa2_pseudo_atom,
    parse_raspa2_force_field,
    save_force_field,
)

DATA_DIR = Path(__file__).parent / "data"


class TestParseRaspa2PseudoAtom:
    """Tests for RASPA2 pseudo atom file parsing."""

    def test_parse_basic(self):
        """Should parse pseudo atoms from a .def file."""
        result = parse_raspa2_pseudo_atom(str(DATA_DIR / "pseudo_atoms.def"))
        assert "PseudoAtoms" in result
        atoms = result["PseudoAtoms"]
        assert len(atoms) == 3

    def test_atom_fields(self):
        """Each atom should have required fields."""
        result = parse_raspa2_pseudo_atom(str(DATA_DIR / "pseudo_atoms.def"))
        atom = result["PseudoAtoms"][0]
        assert "name" in atom
        assert "element" in atom
        assert "mass" in atom
        assert "charge" in atom
        assert "framework" in atom
        assert isinstance(atom["mass"], float)
        assert isinstance(atom["charge"], float)

    def test_trailing_underscore_stripped(self):
        """Atom names with trailing underscore should have it stripped."""
        result = parse_raspa2_pseudo_atom(str(DATA_DIR / "pseudo_atoms.def"))
        names = [a["name"] for a in result["PseudoAtoms"]]
        assert all(not n.endswith("_") for n in names)

    def test_co2_atoms(self):
        """Should correctly parse CO2 pseudo atoms."""
        result = parse_raspa2_pseudo_atom(str(DATA_DIR / "pseudo_atoms.def"))
        atoms_by_name = {a["name"]: a for a in result["PseudoAtoms"]}
        assert "C_co2" in atoms_by_name
        assert "O_co2" in atoms_by_name
        assert atoms_by_name["C_co2"]["charge"] == pytest.approx(0.70)
        assert atoms_by_name["O_co2"]["charge"] == pytest.approx(-0.35)


class TestParseRaspa2ForceField:
    """Tests for RASPA2 force field file parsing."""

    def test_parse_basic(self):
        """Should parse force field from mixing rules file."""
        result = parse_raspa2_force_field(
            str(DATA_DIR / "force_field_mixing_rules.def")
        )
        assert "SelfInteractions" in result
        assert "MixingRule" in result
        assert "TruncationMethod" in result

    def test_interaction_count(self):
        """Should parse the correct number of interactions."""
        result = parse_raspa2_force_field(
            str(DATA_DIR / "force_field_mixing_rules.def")
        )
        assert len(result["SelfInteractions"]) == 3

    def test_mixing_rule(self):
        """Should detect Lorentz-Berthelot mixing rule."""
        result = parse_raspa2_force_field(
            str(DATA_DIR / "force_field_mixing_rules.def")
        )
        assert result["MixingRule"] == "Lorentz-Berthelot"

    def test_interaction_fields(self):
        """Each interaction should have name, type, and parameters."""
        result = parse_raspa2_force_field(
            str(DATA_DIR / "force_field_mixing_rules.def")
        )
        for interaction in result["SelfInteractions"]:
            assert "name" in interaction
            assert "type" in interaction
            assert "parameters" in interaction
            assert len(interaction["parameters"]) == 2

    def test_lennard_jones_params(self):
        """Should correctly parse LJ parameters."""
        result = parse_raspa2_force_field(
            str(DATA_DIR / "force_field_mixing_rules.def")
        )
        interactions = {i["name"]: i for i in result["SelfInteractions"]}
        assert interactions["C_co2"]["parameters"] == [27.0, 2.80]
        assert interactions["O_co2"]["parameters"] == [79.0, 3.05]


class TestSaveForceField:
    """Tests for combined force field output."""

    def test_save_creates_json(self, tmp_path):
        """Should create a force_field.json file."""
        save_force_field(
            str(DATA_DIR / "pseudo_atoms.def"),
            str(DATA_DIR / "force_field_mixing_rules.def"),
            str(tmp_path),
        )
        output = tmp_path / "force_field.json"
        assert output.exists()

    def test_output_valid_json(self, tmp_path):
        """Output should be valid JSON with expected keys."""
        save_force_field(
            str(DATA_DIR / "pseudo_atoms.def"),
            str(DATA_DIR / "force_field_mixing_rules.def"),
            str(tmp_path),
        )
        with open(tmp_path / "force_field.json") as f:
            data = json.load(f)

        assert "PseudoAtoms" in data
        assert "SelfInteractions" in data
        assert "MixingRule" in data
        assert "TruncationMethod" in data

    def test_return_value(self, tmp_path):
        """Should return the combined dict."""
        result = save_force_field(
            str(DATA_DIR / "pseudo_atoms.def"),
            str(DATA_DIR / "force_field_mixing_rules.def"),
            str(tmp_path),
        )
        assert isinstance(result, dict)
        assert len(result["PseudoAtoms"]) == 3
        assert len(result["SelfInteractions"]) == 3
