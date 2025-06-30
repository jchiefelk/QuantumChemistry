import pyscf
import numpy as np
import rdkit.Chem as Chem
import rdkit
from rdkit.Chem import AllChem
from pyscf.geomopt.berny_solver import optimize
from pyscf.hessian import thermo


def compute_in_sto3g(molecule_smiles, molecule_name):
    mol_block = Chem.MolFromSmiles(molecule_smiles)
    mol_block = Chem.AddHs(mol_block)
    AllChem.EmbedMolecule(mol_block)
    xyz = Chem.MolToXYZBlock(mol_block)
    # Get rid of the first two lines in the full XYZ from RDKit
    stripped_xyz = xyz.split("\n", 2)[2:][0]
    mol = pyscf.gto.M(atom=stripped_xyz)
    mol.basis = 'sto-3g' # use a sto-3g basis set
    mol.build()
    # perform a restricted Hartree-Fock (RHF) Calculation
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    # The ground state energy is the total energy obtained from the RHF Calculation
    ground_state_energy = mf.e_tot
    print(f'Ground state energy of {molecule_name} (RHF/STO-3G): {ground_state_energy}')


def compute_in_6_31g(molecule_smiles, molecule_name):
    mol_block = Chem.MolFromSmiles(molecule_smiles)
    mol_block = Chem.AddHs(mol_block)
    AllChem.EmbedMolecule(mol_block)
    xyz = Chem.MolToXYZBlock(mol_block)
    # Get rid of the first two lines in the full XYZ from RDKit
    stripped_xyz = xyz.split("\n", 2)[2:][0]
    mol = pyscf.gto.M(atom=stripped_xyz)
    mol.basis = '6-31g'
    mol.build()
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    ground_state_energy = mf.e_tot
    print(f'Ground state energy of {molecule_name} (RHF/6-31g): {ground_state_energy}')


def optimize_geometry(molecule_smiles, molecule_name):
    mol_block = Chem.MolFromSmiles(molecule_smiles)
    mol_block = Chem.AddHs(mol_block)
    AllChem.EmbedMolecule(mol_block)
    xyz = Chem.MolToXYZBlock(mol_block)
    # Get rid of the first two lines in the full XYZ from RDKit
    stripped_xyz = xyz.split("\n", 2)[2:][0]
    mol = pyscf.gto.M(atom=stripped_xyz)
    mol.basis = 'sto-3g' # use a sto-3g basis set
    mol.build()
    # mf = pyscf.scf.RHF(mol)
    # # mf = scf.RHF(mol)  # For Hartree-Fock
    # # or
    mf = pyscf.dft.RKS(mol)  # For DFT
    mf.xc = 'b3lyp'  # Specify the functional (e.g., B3LYP)

    mol_eq = optimize(mf, maxsteps=100)
    print(mol_eq.atom_coords())
    return mol_eq


def compute_frequencies(molecule_smiles, molecule_name):
    mol_eq = optimize_geometry(molecule_smiles, molecule_name)
    mol = pyscf.gto.M(atom=mol_eq.atom)
    mol.basis = 'sto-3g' # or any other suitable basis set
    mol.verbose = 0
    mol.build()
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    
    return mf


def compute_thermal_analysis(molecule_smiles, molecule_name):
    mf = compute_frequencies(molecule_smiles, molecule_name)
    hessian = mf.Hessian().kernel()
    # Frequency analysis
    freq_info = thermo.harmonic_analysis(mf.mol, hessian)
    # Thermochemistry analysis at 298.15 K and 1 atmospheric pressure
    thermo_info = thermo.thermo(mf, freq_info['freq_au'], 298.15, 101325)
    
    print('Rotation constant')
    print(thermo_info['rot_const'])

    print('Zero-point energy')
    print(thermo_info['ZPE'   ])

    print('Internal energy at 0 K')
    print(thermo_info['E_0K'  ])

    print('Internal energy at 298.15 K')
    print(thermo_info['E_tot' ])

    print('Enthalpy energy at 298.15 K')
    print(thermo_info['H_tot' ])

    print('Gibbs free energy at 298.15 K')
    print(thermo_info['G_tot' ])

    print('Heat capacity at 298.15 K')
    print(thermo_info['Cv_tot'])


def compute_hydronium_ground_state():
    molecule_name = 'Hydronium'
    molecule_smiles = '[OH3+]'
    mol_block = Chem.MolFromSmiles(molecule_smiles)
    mol_block = Chem.AddHs(mol_block)
    AllChem.EmbedMolecule(mol_block)
    xyz = Chem.MolToXYZBlock(mol_block)
    # Get rid of the first two lines in the full XYZ from RDKit
    stripped_xyz = xyz.split("\n", 2)[2:][0]
    mol = pyscf.gto.M(atom=stripped_xyz, spin=1, basis='sto-3g')
    mol.build()

    # Optimize the Geometry
    mf = pyscf.dft.RKS(mol)  # For DFT
    mf.xc = 'b3lyp'
    equilibrated_mol = optimize(mf, maxsteps=100)
    mol = pyscf.gto.M(atom=equilibrated_mol.atom, spin=1, basis='sto-3g')

    # perform a restricted Hartree-Fock (RHF) Calculation
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    # The ground state energy is the total energy obtained from the RHF Calculation
    ground_state_energy = mf.e_tot
    print(f'Ground state energy of {molecule_name} (RHF/STO-3G): {ground_state_energy}')


# compute_hydronium_ground_state()
# molecule_name = 'Methane'
# molecule_smiles = 'C'
# compute_in_sto3g(molecule_smiles, molecule_name)
# compute_thermal_analysis(molecule_smiles, molecule_name)
# compute_frequencies(molecule_smiles, molecule_name)
# compute_in_6_31g(molecule_smiles, molecule_name)
