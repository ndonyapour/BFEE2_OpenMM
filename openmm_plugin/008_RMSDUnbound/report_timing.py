import sys
import os
import os.path as osp
import pickle
import random

import numpy as np
import pickle as pkl

import openmm.app as omma
import openmm as omm
from openmmplumed import PlumedForce
import simtk.unit as unit


import mdtraj as mdj
import parmed as pmd
import time

sys.path.append('../utils')
from BFEE2_CV import RMSD_wall


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

# from wepy, to restart a simulation
GET_STATE_KWARG_DEFAULTS = (('getPositions', True),
                            ('getVelocities', True),
                            ('getForces', True),
                            ('getEnergy', True),
                            ('getParameters', True),
                            ('getParameterDerivatives', True),
                            ('enforcePeriodicBox', True),)
# Platform used for OpenMM which uses different hardware computation
# kernels. Options are: Reference, CPU, OpenCL, CUDA.

PLATFORM = 'CUDA'
PRECISION = 'mixed'
TEMPERATURE = 300.0 * unit.kelvin
FRICTION_COEFFICIENT = 1.0 / unit.picosecond
STEP_SIZE = 0.002 * unit.picoseconds
PRESSURE = 1.0 * unit.atmosphere
VOLUME_MOVE_FREQ = 50

# reporter
NUM_STEPS = 10000000 # 500000 = 1ns
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


#
if not osp.exists(OUTPUTS_PATH):
    os.makedirs(OUTPUTS_PATH)
# the inputs directory and files we need
inputs_dir = osp.realpath(f'./inputs')
prmfile = osp.join(inputs_dir, 'ligandOnly.prmtop')
STAR_CHECKPOINT = osp.join('./outputs_eq','checkpoint_last.chk')

prmtop = omma.amberprmtopfile.AmberPrmtopFile(prmfile)

checkpoint_path = osp.join(OUTPUTS_PATH, CHECKPOINT) # modify based on the simulation
pdb_file = osp.join(inputs_dir, 'ligandOnly.pdb')
pdb = mdj.load_pdb(pdb_file)
# protein and type!="H"'
protein_ligand_idxs = pdb.topology.select('resname "MOL"')

# add disulfide bonds to the topology
#prmtop.topology.createDisulfideBonds(coords.getPositions())

# build the system
system = prmtop.createSystem(nonbondedMethod=omma.PME,
                            nonbondedCutoff=1*unit.nanometer,
                            constraints=omma.HBonds)

# The cpptraj doesn't set the box vectors correctly. set the box vectors
a = unit.Quantity((81.69460 * unit.angstrom, 0.0 * unit.angstrom, 0.0 * unit.angstrom))
b = unit.Quantity((0.0 * unit.angstrom, 81.9153000 * unit.angstrom, 0.0 * unit.angstrom))
c = unit.Quantity((0.0 * unit.angstrom, 0.0 * unit.angstrom, 81.8666 * unit.angstrom))
system.setDefaultPeriodicBoxVectors(a, b, c)

# atm, 300 K, with volume move attempts every 50 steps
barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)
# # add it as a "Force" to the system
system.addForce(barostat)

ligand_idxs = pdb.topology.select('resname "MOL" and type!="H"')
protein_idxs = pdb.topology.select('protein and type!="H"')
ref_pos = omma.pdbfile.PDBFile(pdb_file).getPositions()



# RMSD CV
rmsd_harmonic_wall = RMSD_wall(ref_pos, ligand_idxs,
                            lowerwall=0.0*unit.nanometer,
                            upperwall=0.3*unit.nanometer,
                            force_const=2000*unit.kilojoule_per_mole/unit.nanometer**2)
system.addForce(rmsd_harmonic_wall)


# Metadynamics on RMSD
rmsd_cv = omm.RMSDForce(ref_pos, ligand_idxs)
sigma_rmsd = 0.01
rmsd_bias = omma.metadynamics.BiasVariable(rmsd_cv, minValue=0.0*unit.nanometer, maxValue=0.4*unit.nanometer,
                                           biasWidth=sigma_rmsd*unit.nanometer, periodic=False, gridWidth=400)

bias = 20.0
meta = omma.metadynamics.Metadynamics(system, [rmsd_bias],
                                        TEMPERATURE,
                                        biasFactor=bias,
                                        height=0.5*unit.kilojoules_per_mole,
                                        frequency=HILLS_REPORTER_STEPS)

# make the integrator
integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION)

simulation = omma.Simulation(prmtop.topology, system, integrator, platform, prop)
if osp.exists(STAR_CHECKPOINT):
    print("Start from checkpoint")
    simulation.loadCheckpoint(STAR_CHECKPOINT)
else:
    print("can not find the checkpoint")
# make the integrator
report_timing(meta, simulation, "Run time for this step:")