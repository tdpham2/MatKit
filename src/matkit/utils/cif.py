from __future__ import annotations


def sanitize_cif_stem(stem: str) -> str:
    """Return a gRASPA-safe CIF stem.

    gRASPA's input parser treats the first ``.`` in ``FrameworkName`` as
    the extension separator, so stems with extra periods (e.g.
    ``str_m5_o11_o18_sra_sym.22``) get truncated and the framework file
    cannot be located. Replace every ``.`` with ``_`` to make the stem
    safe; stems without ``.`` are returned unchanged.
    """
    return stem.replace(".", "_") if "." in stem else stem
