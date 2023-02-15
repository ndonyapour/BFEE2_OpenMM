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
from openmmtools.integrators import GradientDescentMinimizationIntegrator


import mdtraj as mdj
import parmed as pmd
import time

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
DCD_REPORT_STEPS = 5000
CHECKPOINT_REPORTER_STEPS =  5000
OUTPUTS_PATH = osp.realpath(f'outputs')
SIM_TRAJ = 'traj.dcd'
CHECKPOINT = 'checkpoint.chk'
CHECKPOINT_LAST = 'checkpoint_last.chk'
SYSTEM_FILE = 'system.pkl'
OMM_STATE_FILE = 'state.pkl'

#
if not osp.exists(OUTPUTS_PATH):
    os.makedirs(OUTPUTS_PATH)
# the inputs directory and files we need
inputs_dir = osp.realpath(f'../inputs')
prmfile = osp.realpath('../000_eq/outputs_namd/complex.parm7')
# coodsfile = osp.join(inputs_dir, 'complex.rst7')
coords = pmd.namd.namdbinfiles.NamdBinCoor.read('../000_eq/outputs_namd/eq.coor').coordinates[0] / 10
velocities = pmd.namd.namdbinfiles.NamdBinVel.read('../000_eq/outputs_namd/eq.vel').velocities[0]
prmtop = pmd.load_file(prmfile)
# prmtop = omma.amberprmtopfile.AmberPrmtopFile(prmfile)
# coords = omma.amberinpcrdfile.AmberInpcrdFile(coodsfile).getPositions()
# coords = omma.pdbfile.PDBFile('inputs/complex_namd.pdb').getPositions()

plumed_file = osp.realpath('plumed.dat')
checkpoint_path = osp.join(OUTPUTS_PATH, CHECKPOINT) # modify based on the simulation



# add disulfide bonds to the topology
#prmtop.topology.createDisulfideBonds(coords.getPositions())

# build the system
system = prmtop.createSystem(nonbondedMethod=omma.PME,
                            nonbondedCutoff=1*unit.nanometer,
                            constraints=omma.HBonds)

# atm, 300 K, with volume move attempts every 50 steps
barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)
# # add it as a "Force" to the system
system.addForce(barostat)

# make the integrator
integrator = GradientDescentMinimizationIntegrator(initial_step_size=0.0001)


platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION)

simulation = omma.Simulation(prmtop.topology, system, integrator, platform, prop)

simulation.context.setPositions(coords)

print("Before minimization = ", pmd.openmm.energy_decomposition(prmtop, simulation.context, nrg=unit.kilojoules_per_mole))
simulation.step(10)
print("After minimization = ", pmd.openmm.energy_decomposition(prmtop, simulation.context, nrg=unit.kilojoules_per_mole))
