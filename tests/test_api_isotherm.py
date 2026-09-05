"""Protect the physical axes and units used by single-isotherm plots."""

import math

import pytest

from matkit.plot.parsers import parse_single_isotherm


@pytest.mark.parametrize(
    "data,unit,pressures,uptakes",
    [
        (
            {
                "2bar_298K": {"uptake": 2},
                "100000Pa_298K": {"uptake": 1},
            },
            "bar",
            [1, 2],
            [1, 2],
        ),
        (
            {
                "100000Pa_298K": {"uptake": 1},
                "2bar_298K": {"uptake": 2},
            },
            "Pa",
            [100000, 200000],
            [1, 2],
        ),
        (
            {
                "0bar_298K": {"uptake": 0},
                "0.1Pa_298.0K": {"uptake": 1},
            },
            "bar",
            [0, 0.000001],
            [0, 1],
        ),
    ],
)
def test_pressure_units_are_converted_before_sorting(
    data, unit, pressures, uptakes
):
    result = parse_single_isotherm(data)
    assert result["pressure_unit"] == unit
    assert result["pressures"] == pytest.approx(pressures)
    assert result["uptakes"] == uptakes
    assert result["temperature"] == 298


@pytest.mark.parametrize(
    "first,second",
    [
        ("1bar_298K", "100000Pa_298K"),
        ("0.29bar_298K", "29000Pa_298K"),
        ("0.1bar_298K", "0.10bar_298.0K"),
        ("0bar_298K", "0Pa_298K"),
    ],
)
def test_duplicate_physical_pressures_are_rejected(first, second):
    with pytest.raises(ValueError, match="Duplicate physical pressure"):
        parse_single_isotherm({first: {"uptake": 1}, second: {"uptake": 2}})


def test_mixed_temperatures_are_rejected():
    with pytest.raises(ValueError, match="mix temperatures"):
        parse_single_isotherm(
            {
                "1bar_298K": {"uptake": 1},
                "2bar_350K": {"uptake": 2},
            }
        )


@pytest.mark.parametrize("first,second", [("mol/kg", "mg/g"), ("mg/g", "g/L")])
def test_mixed_uptake_units_are_rejected(first, second):
    with pytest.raises(ValueError, match="mix uptake units"):
        parse_single_isotherm(
            {
                "1bar_298K": {"uptake": 1, "unit": first},
                "2bar_298K": {"uptake": 2, "unit": second},
            }
        )


@pytest.mark.parametrize("second_unit", ["kcal/mol", None])
def test_mixed_or_unknown_heat_units_are_rejected(second_unit):
    with pytest.raises(ValueError, match="mix heat-of-adsorption units"):
        parse_single_isotherm(
            {
                "1bar_298K": {"uptake": 1, "qst": 10, "qst_unit": "kJ/mol"},
                "2bar_298K": {"uptake": 2, "qst": 20, "qst_unit": second_unit},
            }
        )


def test_missing_heat_does_not_hide_units_of_available_heat():
    result = parse_single_isotherm(
        {
            "1bar_298K": {"uptake": 1},
            "2bar_298K": {"uptake": 2, "qst": 10, "qst_unit": "kJ/mol"},
        }
    )
    assert result["qst_unit"] == "kJ/mol"
    assert math.isnan(result["qst"][0])
    assert result["qst"][1] == 10
