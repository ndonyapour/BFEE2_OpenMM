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


def report_timing(system, positions, description):
    """Report timing on all available platforms."""
    timestep = 2.0*unit.femtoseconds
    nsteps = 5000
    for platform_name in ['CUDA']:
        platform = omm.Platform.getPlatformByName(platform_name)

        integrator = omm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, timestep)
        context = omm.Context(system, integrator, platform)
        context.setPositions(positions)
        # Warm up the integrator
        integrator.step(10)
        # Time integration
        initial_time = time.time()
        integrator.step(nsteps)
        final_time = time.time()
        elapsed_time = (final_time - initial_time) * unit.seconds
        ns_per_day = nsteps * timestep / elapsed_time / (unit.nanoseconds / unit.day)
        print('%64s : %16s : %8.3f ns/day' % (description, platform_name, ns_per_day))
        del context, integrator

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
NUM_STEPS = 10 #5000000 # 500000 = 1ns
DCD_REPORT_STEPS = 5000
CHECKPOINT_REPORTER_STEPS =  5000
LOG_REPORTER_STEPS = 500
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
prmfile = osp.join(inputs_dir, 'complex.prmtop')
coodsfile = osp.join(inputs_dir, 'complex.rst7')

prmtop = omma.amberprmtopfile.AmberPrmtopFile(prmfile)
coords = omma.amberinpcrdfile.AmberInpcrdFile(coodsfile).getPositions()


plumed_file = osp.realpath('plumed.dat')
checkpoint_path = osp.join(OUTPUTS_PATH, CHECKPOINT) # modify based on the simulation


# build the system
system = prmtop.createSystem(nonbondedMethod=omma.PME,
                            nonbondedCutoff=1*unit.nanometer,
                            constraints=omma.HBonds)

# atm, 300 K, with volume move attempts every 50 steps
barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)
system.addForce(barostat)

# add Plumed
with open(plumed_file, 'r') as file:
    script = file.read()
system.addForce(PlumedForce(script))

report_timing(system, coords, "000_Equilibration step")

# make the integrator
integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION)

simulation = omma.Simulation(prmtop.topology, system, integrator, platform, prop)

print("New Simulation")
simulation.context.setPositions(coords)

simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH, SIM_TRAJ),
                                                                DCD_REPORT_STEPS))
                                                                #atomSubset=protein_ligand_idxs))

simulation.reporters.append(omma.CheckpointReporter(checkpoint_path,
                                                    CHECKPOINT_REPORTER_STEPS))

simulation.reporters.append(
    omma.StateDataReporter(
        "log",
        LOG_REPORTER_STEPS,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        volume=True,
        temperature=True,
        totalSteps=True,
        separator=" ",
    )
)

print("Start Simulation")
start_time = time.time()
simulation.step(NUM_STEPS)
end_time = time.time()
print("End Simulation")
print(f"Run time = {np.round(end_time - start_time, 3)}s")
simulation_time = round((STEP_SIZE * NUM_STEPS).value_in_unit(unit.nanoseconds),
                           2)
print(f"Simulation time: {simulation_time}ns")
simulation.saveCheckpoint(osp.join(OUTPUTS_PATH, CHECKPOINT_LAST))

# save final state and system
get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
omm_state = simulation.context.getState(**get_state_kwargs)
# save the pkl files to the inputs dir
with open(osp.join(OUTPUTS_PATH, SYSTEM_FILE), 'wb') as wfile:
    pkl.dump(system, wfile)

with open(osp.join(OUTPUTS_PATH, OMM_STATE_FILE), 'wb') as wfile:
    pkl.dump(omm_state, wfile)
print('Done making pkls. Check inputs dir for them!')