import random
import shutil
from pathlib import Path
from typing import List


def sample_cifs(
    n_cifs: int,
    path_to_cifs: str,
    copy: bool = False,
    outpath: str = None,
    seed: int = 2025,
) -> List[str]:
    """
    Randomly sample CIF files from a directory.

    Parameters:
        n_cifs (int): Number of CIFs to sample.
        path_to_cifs (str): Path to the directory containing CIF files.
        copy (bool): Whether to copy sampled CIFs to a new location.
        outpath (str): Destination directory for copied files
                                 (required if copy=True).
        seed (int): Seed value for reproducibility.
    Returns:
        List[str]: Filenames of the sampled CIFs.
    """
    path_to_cifs = Path(path_to_cifs)
    if not path_to_cifs.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {path_to_cifs}"
        )

    cifs = [f.name for f in path_to_cifs.glob("*.cif")]
    if len(cifs) < n_cifs:
        raise ValueError(
            f"Requested {n_cifs} CIFs but only {len(cifs)} available."
        )
    random.seed(seed)
    random.shuffle(cifs)
    selected_cifs = cifs[:n_cifs]

    if not copy:
        return selected_cifs

    if outpath is None:
        raise ValueError("Copying requested but no output path was specified.")

    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    for cif in selected_cifs:
        src = path_to_cifs / cif
        dst = outpath / cif
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            raise RuntimeError(f"Failed to copy {src} to {dst}: {e}")

    return selected_cifs
