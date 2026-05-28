from __future__ import annotations

import shutil
from pathlib import Path


def copy_template(template_dir: Path, output_dir: Path) -> None:
    """Copy all files and directories from a template directory.

    Args:
        template_dir: Source directory containing template files.
        output_dir: Destination directory (created if needed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in template_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, output_dir, dirs_exist_ok=True)
        else:
            shutil.copy2(item, output_dir)


def render_template(file_path: Path, substitutions: dict[str, str]) -> None:
    """Apply placeholder substitutions to a file in-place.

    Reads the file, replaces each key in *substitutions* with its
    value using ``str.replace``, and writes the result back.

    Args:
        file_path: Path to the file to modify.
        substitutions: Mapping of placeholder strings to their
            replacement values.
    """
    content = file_path.read_text()
    for key, val in substitutions.items():
        content = content.replace(key, val)
    file_path.write_text(content)
