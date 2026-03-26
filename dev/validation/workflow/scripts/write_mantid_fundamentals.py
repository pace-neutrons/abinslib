from tempfile import TemporaryDirectory

from euphonic import Quantity, Spectrum1D
import mantid.simpleapi
from snakemake.script import snakemake

import abins.parameters

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

    mantid.simpleapi.Abins(**abins_kwargs)

workspace = mantid.simpleapi.mtd[f"{WORKSPACE}_total"]
energy = workspace.getAxis(0).extractValues()
bin_width = energy[1] - energy[0]
energy_unit = workspace.getAxis(0).getUnit().symbol().ascii()
intensity = workspace.extractY()[0] / bin_width

spectrum = Spectrum1D(
    Quantity(energy, str(energy_unit)),
    Quantity(intensity, f"barn / {energy_unit}"),
    metadata={"temperature": str(abins_kwargs["TemperatureInKelvin"])},
)

spectrum.to_json_file(snakemake.output[0])
