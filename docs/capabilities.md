# Capability inventory

This inventory distinguishes implemented interfaces from scientific validation.
No GPU or external-engine capability has been promoted by CPU fixtures. The
unified API remains experimental; record real execution for each capability,
model, and environment before promotion.

| Interface | Implemented capability | Environment | Evidence/status |
| --- | --- | --- | --- |
| Unified direct MACE | Energy, requested forces/stress, fixed-cell relaxation, sequential batches | MACE + compatible CPU/CUDA stack; model-dependent properties/species | CPU adapter contracts; experimental |
| Unified Rootstock | Energy, requested forces/stress, fixed-cell relaxation, sequential batches | Rootstock client and separately installed deployment | CPU adapter contracts; worker evidence required |
| Unified ALCHEMI MACE | Energy, requested forces/stress, fixed-cell FIRE, native batches | Compatible ALCHEMI/CUDA environment | Mocked native contracts; GPU evidence required |
| Unified Zeo++ | Diameter, area, volume, PSD, channels | `network` binary and radii definitions | Parser/subprocess fixtures; real execution required |
| Unified gRASPA CUDA | Pure-component preparation, execution, absolute uptake and heat parsing | Charged periodic CIF; templates; CUDA executable for execution | Synthetic output/subprocess fixtures; real execution required |
| Legacy gRASPA/pygRASPA | Pure/mixture and grid setup; existing parsers; pygRASPA reference-energy helper | Core for preparation; engine-specific environment for execution | Setup/parser fixtures; mixtures outside unified result contract |
| Legacy gRASPA SYCL | Setup and parsing | Core for preparation; Intel GPU environment for execution | Cutoff regressions; Aurora recipe; execution evidence required |
| Legacy RASPA2 | Setup and parsing | Core for preparation; RASPA2 for execution | Cutoff and success-reporting regressions; execution evidence required |
| RASPA3 | Force-field conversion only | Core | Conversion fixtures; simulation execution not implemented |
| Legacy MACE optimization | Geometry, cell, sequential geometry/cell optimization | MACE with required forces/stress | Existing interface; outside unified validation contract |
| Legacy UMA | Single point, geometry/cell optimization, MD and batch optimization | FAIRChem/UMA installation | Existing interfaces; outside unified validation contract |
| PACMOF2 | Charge-prediction wrapper | PACMOF2 installation | Existing interface; per-output scientific validation not yet unified |
| Structure utilities/ToBaCCo | Solvent removal, sampling, linker/conversion helpers | Core; optional RDKit/Open Babel as applicable | Legacy compatibility; future structural operations belong in MOFforge |
| Isotherm plotting | Single/mixture plots and selectivity from existing data | Matplotlib extra | Parser/plot fixtures; does not establish input scientific accuracy |
| ORCA | Stub | Not applicable | No supported execution capability |
| CLI/MCP | Unified requests, results, artifacts; bounded stdio tools | Core CLI; optional MCP SDK 2 | Deterministic subprocess and local stdio integration tests |

Availability in `matkit capabilities` is a caller-side installation check, not a
model suitability or GPU compatibility claim. Preparation needs no engine
binary. Generic energy evaluation needs only its requested properties;
relaxation needs forces, and cell optimization is not advertised by the unified
API. Native ALCHEMI rejects unsupported atom arrays/constraints. Model aliases
whose content cannot be resolved are identified as such in provenance.

Promotion requires interface documentation, licensed reference fixtures,
failure tests, reproducible installation, and reviewed real execution. Numerical
parity uses matching checkpoints/settings; scientific accuracy requires
independent reference data. Record benchmark startup, warm execution, throughput,
memory, and failures separately.
