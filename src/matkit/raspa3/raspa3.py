import json
import os


def parse_raspa2_pseudo_atom(filepath):
    pseudo_atoms = []

    with open(filepath, "r") as file:
        lines = file.readlines()

    # Skip first two lines (header)
    data_lines = lines[2:]

    for line in data_lines:
        if line.strip() == "" or line.strip().startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 14:
            continue  # skip malformed lines

        raw_name = parts[0]
        name = raw_name[:-1] if raw_name.endswith("_") else raw_name
        framework = parts[1].lower() in ["yes", "true", "1"]
        print_as = parts[2]
        element = parts[3]
        mass = float(parts[5])
        charge = float(parts[6])

        atom = {
            "name": name,
            "framework": framework,
            "print_to_output": True,
            "element": element,
            "print_as": print_as,
            "mass": mass,
            "charge": charge,
        }

        pseudo_atoms.append(atom)

    return {"PseudoAtoms": pseudo_atoms}


def parse_raspa2_force_field(filepath, default_source="unknown"):
    interactions = []
    mixing_rule = None
    truncation_method = None

    with open(filepath, "r") as file:
        lines = file.readlines()

    # Clean lines: remove blank lines and comments
    data_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    if len(data_lines) < 3:
        raise ValueError("File too short or missing required sections")

    # First line is the truncation method
    truncation_method = data_lines[0]

    # Third line is number of interactions
    try:
        interaction_count = int(data_lines[2])
    except ValueError:
        raise ValueError(
            f"Expected number of interactions on line 3, got: {data_lines[2]}"
        )

    # Extract interaction lines
    interaction_lines = data_lines[3 : 3 + interaction_count]
    for line in interaction_lines:
        parts = line.split()
        if len(parts) < 4:
            continue  # Skip malformed lines

        raw_name = parts[0]
        name = raw_name[:-1] if raw_name.endswith("_") else raw_name
        interaction_type = parts[1]
        epsilon = float(parts[2])
        sigma = float(parts[3])

        interaction = {
            "name": name,
            "type": interaction_type,
            "parameters": [epsilon, sigma],
            "source": default_source,
        }
        interactions.append(interaction)

    # Look for mixing rule in remaining lines
    remaining_lines = data_lines[3 + interaction_count :]
    for line in remaining_lines:
        if "Lorentz-Berthelot" in line:
            mixing_rule = "Lorentz-Berthelot"
            break
        elif "geometric" in line.lower():
            mixing_rule = "geometric"
            break

    if not mixing_rule:
        raise ValueError("Mixing rule not found in the file")

    return {
        "SelfInteractions": interactions,
        "MixingRule": mixing_rule,
        "TruncationMethod": truncation_method,
    }


def save_force_field(pseudo_fp, force_field_fp, outpath):
    pseudo_atoms = parse_raspa2_pseudo_atom(pseudo_fp)
    force_field = parse_raspa2_force_field(force_field_fp)

    combined = {
        "PseudoAtoms": pseudo_atoms["PseudoAtoms"],
        "SelfInteractions": force_field["SelfInteractions"],
        "MixingRule": force_field["MixingRule"],
        "TruncationMethod": force_field["TruncationMethod"],
    }
    output_file = os.path.join(outpath, "force_field.json")
    with open(output_file, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved combined force field to {output_file}")
    return combined
