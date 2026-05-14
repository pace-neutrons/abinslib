from tempfile import TemporaryDirectory

from euphonic import Quantity, Spectrum1D
import mantid.simpleapi
from snakemake.script import snakemake

import abins.parameters  # isort:skip


WORKSPACE = "wrk"

if snakemake.params["raw"]:
    # Disable instrumental resolution broadening
    abins.parameters.sampling["broadening_scheme"] = "none"

if snakemake.params.get("isotropic", False):
    # Use a simpler fully-isotropic intensity model
    abins.parameters.development["isotropic_fundamentals"] = True

with TemporaryDirectory() as tmpdir:
    abins_kwargs = {
        "VibrationalOrPhononFile": snakemake.input[0],
        "AbInitioProgram": "JSON",
        "OutputWorkspace": WORKSPACE,
        "CacheDirectory": tmpdir,
    } | snakemake.params["abins_kwargs"]
    abins_kwargs["QuantumOrderEventsNumber"] = "2"

    mantid.simpleapi.Abins(**abins_kwargs)

def _is_o2(ws: str) -> bool:
    return '_quantum_event_2' in ws

o2_workspace_names = filter(_is_o2, mantid.simpleapi.mtd.getObjectNames())
o2_workspaces = [mantid.simpleapi.mtd[name] for name in o2_workspace_names]

energy = o2_workspaces[0].getAxis(0).extractValues()
bin_width = energy[1] - energy[0]
energy_unit = o2_workspaces[0].getAxis(0).getUnit().symbol().ascii()

intensity = sum(workspace.extractY()[0] for workspace in o2_workspaces)
intensity /= bin_width

spectrum = Spectrum1D(
    Quantity(energy, str(energy_unit)),
    Quantity(intensity, f"barn / {energy_unit}"),
    metadata={"temperature": str(abins_kwargs["TemperatureInKelvin"])},
)

spectrum.to_json_file(snakemake.output[0])
