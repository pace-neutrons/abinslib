from euphonic import Quantity, Spectrum1D
from euphonic.plot import plot_1d_to_axis
import matplotlib.pyplot as plt
from snakemake.script import snakemake

width = Quantity(1, "meV")

abinslib_data = Spectrum1D.from_json_file(snakemake.input[0]).broaden(width)
mantid_data = Spectrum1D.from_json_file(snakemake.input[1]).broaden(width)

fig, ax = plt.subplots()

plot_1d_to_axis(abinslib_data, ax, label="q_scaling_isotropic_incoherent_spectra")
plot_1d_to_axis(
    mantid_data, ax, linestyle="--", label="Mantid-Abins isotropic fundamentals"
)
ax.legend()
ax.set_title(
    "Ethanol isotropic fundamentals:\n134.99° $E_f = 32 "
    f"\\text{{cm}}^{{-1}}$\n{width:~P} broadening"
)
ax.set_xlim(0, 200)
ax.set_xlabel("Energy transfer / meV")
ax.set_ylabel("$S(\omega)$ / barn meV$^{-1}$")

fig.tight_layout()
fig.savefig(snakemake.output[0])
