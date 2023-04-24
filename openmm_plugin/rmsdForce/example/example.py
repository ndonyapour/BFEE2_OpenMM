import os
import os.path as osp

import openmm.app as omma
import openmm as omm
import simtk.unit as unit

import mdtraj as mdj
import parmed as pmd

DEVICEINDEX = '0,1'
PLATFORM = 'CUDA'
PRECISION = 'mixed'
TEMPERATURE = 300.0 * unit.kelvin
FRICTION_COEFFICIENT = 1.0 / unit.picosecond
STEP_SIZE = 0.002 * unit.picoseconds
PRESSURE = 1.0 * unit.atmosphere
VOLUME_MOVE_FREQ = 50
NUM_STEPS = 500 # 500000 = 1ns

inputs_dir = osp.realpath('.')
prmfile = osp.join(inputs_dir, 'complex.prmtop')
pdb_file = osp.join(inputs_dir, 'complex.pdb')

positions = omma.pdbfile.PDBFile(pdb_file).getPositions()
prmtop = omma.amberprmtopfile.AmberPrmtopFile(prmfile)

# build the system
system = prmtop.createSystem(nonbondedMethod=omma.PME,
                            nonbondedCutoff=1*unit.nanometer,
                            constraints=omma.HBonds)

# 1 atm, 300 K, with volume move attempts every 50 steps
barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)
system.addForce(barostat)


ligand_idxs = mdj.load_pdb(pdb_file).topology.select('resname "MOL" and type!="H"')
protein_idxs = mdj.load_pdb(pdb_file).topology.select('protein and type!="H"')

r = omm.CustomCentroidBondForce(2, 'distance(g1,g2)')
r.addGroup(ligand_idxs)
r.addGroup(protein_idxs)
r.addBond([0, 1], [])

cvforce = omm.CustomCVForce('r')
cvforce.addCollectiveVariable('r', r)
system.addForce(cvforce)

platform = omm.Platform.getPlatformByName(PLATFORM)
integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)
properties = dict(Precision=PRECISION, DeviceIndex=DEVICEINDEX)
simulation = omma.Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(positions)
simulation.step(NUM_STEPS)

print("Simulation run successfully")
