"""Calculate phonons with PET-MAD and write force constants."""

from ase import Atoms
import ase.build
import numpy as np
from phonopy import Phonopy
from phonopy.file_IO import write_force_constants_to_hdf5
from phonopy.structure.atoms import PhonopyAtoms
from snakemake.script import snakemake
from tqdm import tqdm
from upet.calculator import UPETCalculator


def phonopy_from_ase(atoms: Atoms) -> PhonopyAtoms:
    """Convert ASE Atoms to PhonopyAtoms object."""
    return PhonopyAtoms(
        symbols=atoms.symbols,
        cell=atoms.cell,
        scaled_positions=atoms.get_scaled_positions(),
    )


def ase_from_phonopy(phonopy_atoms: PhonopyAtoms) -> Atoms:
    """Convert PhonopyAtoms to ASE Atoms object."""
    return Atoms(
        phonopy_atoms.symbols,
        cell=phonopy_atoms.cell,
        scaled_positions=phonopy_atoms.scaled_positions,
        pbc=True,
    )


def run_phonons(
    atoms: Atoms,
    calc: UPETCalculator,
    supercell: np.ndarray,
    symprec: float = 1e-4,
    label: str = "atoms",
    distance: float = 1e-3,
) -> None:
    """Generate force constants with Phonopy and write to yaml+hdf5."""
    # Phonopy uses its own ASE-like structure container
    phonopy = Phonopy(
        phonopy_from_ase(atoms), supercell_matrix=supercell, symprec=symprec
    )
    phonopy.generate_displacements(distance=distance)

    def _get_forces(atoms: Atoms) -> np.ndarray:
        atoms.calc = calc
        return atoms.get_forces()

    all_forces = [
        _get_forces(ase_from_phonopy(displaced_supercell))
        for displaced_supercell in tqdm(phonopy.supercells_with_displacements)
    ]

    phonopy.forces = all_forces
    phonopy.produce_force_constants()

    phonopy_file = f"results/{label}-phonopy.yaml"
    hdf5_file = f"results/{label}-force_constants.hdf5"
    print(f"Saving to files {phonopy_file} and {hdf5_file} ...")
    phonopy.save(filename=phonopy_file, settings={"force_constants": False})

    write_force_constants_to_hdf5(phonopy.force_constants, filename=hdf5_file)


def main():  # noqa: D103
    calculator = UPETCalculator(**snakemake.params.calc_params)

    atoms = ase.io.read(snakemake.input[0])
    run_phonons(
        atoms,
        calculator,
        supercell=[3, 3, 3],
        label=snakemake.params["label"],
        distance=snakemake.params.get("displacement", 1e-3),
    )


if __name__ == "__main__":
    main()
