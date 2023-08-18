import sys
import os
import os.path as osp
import pickle
import random

import numpy as np
import pickle as pkl

import openmm.app as omma
import openmm as omm
import simtk.unit as unit

import mdtraj as mdj
import parmed as pmd
import time

sys.path.append('../utils')
from BFEE2_CV import *
from Polaranglesplugin import PolaranglesForce

def report_timing(meta, simulation, description):
    """Report timing on all available platforms."""
    timestep = 2.0*unit.femtoseconds
    nsteps = 5000
    for platform_name in ['CUDA']:
        meta.step(simulation, 10)
        # Warm up the integrator
        # Time integration
        initial_time = time.time()
        meta.step(simulation, nsteps)
        final_time = time.time()
        elapsed_time = (final_time - initial_time) * unit.seconds
        ns_per_day = nsteps * timestep / elapsed_time / (unit.nanoseconds / unit.day)
        print('\n *************%64s : %16s : %8.3f ns/day ***************\n' % (description, platform_name, ns_per_day))

PLATFORM = 'CUDA'
PRECISION = 'mixed'
TEMPERATURE = 300.0 * unit.kelvin
FRICTION_COEFFICIENT = 1.0 / unit.picosecond
STEP_SIZE = 0.002 * unit.picoseconds
PRESSURE = 1.0 * unit.atmosphere
VOLUME_MOVE_FREQ = 50

# reporter
NUM_STEPS = 5000000 # 500000 = 1ns   #5000000
REPORTER_STEPS = 1000
DCD_REPORTER_STEPSS = 50000
HILLS_REPORTER_STEPS = 500
COLVAR_REPORTER_STEPS = 5000
CHECKPOINT_REPORTER_STEPS =  5000
LOG_REPORTER_STEPS = 50000
OUTPUTS_PATH = osp.realpath(f'outputs')
SIM_TRAJ = 'traj.dcd'
CHECKPOINT = 'checkpoint.chk'
CHECKPOINT_LAST = 'checkpoint_last.chk'
SYSTEM_FILE = 'system.pkl'
OMM_STATE_FILE = 'state.pkl'
LOG_FILE = 'log'
STAR_CHECKPOINT = '../../openmm_plumed/000_eq/outputs/checkpoint_last.chk'

#
if not osp.exists(OUTPUTS_PATH):
    os.makedirs(OUTPUTS_PATH)

# the inputs directory and files we need
inputs_dir = osp.realpath(f'../../openmm_plumed/inputs')

prmfile = osp.join(inputs_dir, 'complex.prmtop')
prmtop = omma.amberprmtopfile.AmberPrmtopFile(prmfile)

checkpoint_path = osp.join(OUTPUTS_PATH, CHECKPOINT) # modify based on the simulation

pdb_file = osp.join(inputs_dir, 'complex_bfee2.pdb')
pdb = mdj.load_pdb(pdb_file)


# protein and type!="H"'
protein_ligand_idxs = pdb.topology.select('protein or resname "MOL"')
ligand_idxs = pdb.topology.select('resname "MOL" and type!="H"')
protein_idxs = pdb.topology.select('protein and type!="H"')
ref_pos = omma.pdbfile.PDBFile(pdb_file).getPositions()


system = prmtop.createSystem(nonbondedMethod=omma.PME,
                            nonbondedCutoff=1*unit.nanometer,
                            constraints=omma.HBonds)


# atm, 300 K, with volume move attempts every 50 steps
barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)
# # add it as a "Force" to the system
system.addForce(barostat)


# Translation restraint on protein
com = mdj.compute_center_of_mass(pdb, select='protein and type!="H"')
dummy_atom_pos = omm.vec3.Vec3(*com[0])*unit.nanometers
translation_res = Translation_restraint(protein_idxs, dummy_atom_pos,
                                 force_const=41840*unit.kilojoule_per_mole/unit.nanometer**2) #41840
system.addForce(translation_res)

# Orientaion restraint
q_centers = [1.0, 0.0, 0.0, 0.0]
q_force_consts = [8368*unit.kilojoule_per_mole/unit.nanometer**2 for _ in range(4)]
orientaion_res = Orientaion_restraint(ref_pos, protein_idxs.tolist(), q_centers, q_force_consts)
system.addForce(orientaion_res)

# harmonic restraint on ligand rmsd
rmsd_res = RMSD_harmonic(ref_pos, ligand_idxs.tolist(), center=0.0*unit.nanometer,
                         force_const=4184*unit.kilojoule_per_mole/unit.nanometer**2) # 4184

system.addForce(rmsd_res)

# Euler Theta CV

# harmonic restraint on Euler theta
eulertheta_res = EulerAngle_harmonic(ref_pos, ligand_idxs.tolist(),
                                     protein_idxs.tolist(),
                                     center=4.57,
                                     angle="Theta",
                                     force_const=0.4184)
system.addForce(eulertheta_res)

# harmonic restraint on Euler phi
eulerphi_res = EulerAngle_harmonic(ref_pos, ligand_idxs.tolist(),
                                     protein_idxs.tolist(),
                                     center=-20.24,
                                     angle="Phi",
                                     force_const=0.4184)
system.addForce(eulerphi_res)

# harmonic restraint on Euler psi
eulerpsi_res = EulerAngle_harmonic(ref_pos, ligand_idxs.tolist(),
                                     protein_idxs.tolist(),
                                     center=13.53,
                                     angle="Psi",
                                     force_const=0.4184)
system.addForce(eulerpsi_res)

harmonic_wall = PolarAngle_wall(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(),
                                           angle="Theta",
                                           lowerwall=47.0, # fails when passing with units -15.0* unit.degree
                                           upperwall=87.0,
                                           force_const=100)#*unit.kilojoule_per_mole/unit.degree**2)

system.addForce(harmonic_wall)
# # using modefied version from biosimspace
sigma = 0.5
cv = PolaranglesForce(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(), "Theta")
bias = omma.metadynamics.BiasVariable(cv, minValue=42.0, maxValue=93.0,
                                      biasWidth=sigma, periodic=False, gridWidth=400)
bias_factor = 15.0
meta = omma.metadynamics.Metadynamics(system, [bias],
                    TEMPERATURE,
                    biasFactor=bias_factor,
                    height=0.01*unit.kilojoules_per_mole,
                    frequency=HILLS_REPORTER_STEPS)

integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION)

# for i, f in enumerate(system.getForces()):
#     f.setForceGroup(i)

simulation = omma.Simulation(prmtop.topology, system, integrator, platform, prop)
if osp.exists(STAR_CHECKPOINT):
    print(f"Start Simulation from checkpoint {STAR_CHECKPOINT}")
    simulation.loadCheckpoint(STAR_CHECKPOINT)
else:
    print("Can not find the checkpoint")

report_timing(meta, simulation, "Run time for this step:")