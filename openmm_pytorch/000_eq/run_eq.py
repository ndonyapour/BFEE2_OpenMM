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
NUM_STEPS = 1000 # 500000 = 1ns # test
DCD_REPORT_STEPS = 500000
CHECKPOINT_REPORTER_STEPS =  5000
LOG_REPORTER_STEPS = 500
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

inputs_dir = osp.realpath(f'../../openmm_plumed/inputs')
prmfile = osp.join(inputs_dir, 'complex.prmtop')

# for openmm simulations (0,0,0) is the left corner of box, cerntering molceules to (0, 0, 0) cuases proplems.
#coodsfile = osp.join(inputs_dir, 'complex.rst7')
plumed_file = osp.realpath('plumed.dat')
coords = omma.pdbfile.PDBFile(osp.join(inputs_dir, 'complex_bfee2.pdb')).getPositions()


prmtop = omma.amberprmtopfile.AmberPrmtopFile(prmfile)
# import ipdb
# ipdb.set_trace()

#coords = omma.amberinpcrdfile.AmberInpcrdFile(coodsfile).getPositions()
checkpoint_path = osp.join(OUTPUTS_PATH, CHECKPOINT)


# build the system
system = prmtop.createSystem(nonbondedMethod=omma.PME,
                            nonbondedCutoff=1*unit.nanometer,
                            constraints=omma.HBonds)


# add a dummy atom (ref protein COM) and set its mass to zero
# find the neighbors
# pos = coords.values_
# dist = np.linalg.norm(pos - np.array([4.27077094, 3.93215937, 3.84423549]))
# import ipdb
# ipdb.set_trace()

# dummy_atom_idx = system.addParticle(0.0)
# coords.append(omm.vec3.Vec3(4.27077094, 3.93215937, 3.84423549)*unit.nanometers)


# # the same particle should be added to the nonbonded forces
# nonbonded = [f for f in system.getForces() if isinstance(f, omm.NonbondedForce)]
# for force in nonbonded:
#     # charge, sigma, epsilon
#     force.addParticle(0.0, 1.0, 0.0)
#     for i in range(6000):
#         import ipdb
#         ipdb.set_trace()
#         force.addException(i, dummy_atom_idx, 0.0, 1.0, 0.0)

    # for i in range(dummy_atom_idx):
    #     force.addException(i, dummy_atom_idx, 0.0, 1.0, 0.0)

# import ipdb
# ipdb.set_trace()
# atm, 300 K, with volume move attempts every 50 steps
barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)
system.addForce(barostat)

# add Plumed
with open(plumed_file, 'r') as file:
    script = file.read()
#system.addForce(PlumedForce(script))

# make the integrator
integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION)

simulation = omma.Simulation(prmtop.topology, system, integrator, platform, prop)



print("New Simulation")
simulation.context.setPositions(coords)

get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
omm_state = simulation.context.getState(**get_state_kwargs)
positions = omm_state.getPositions(asNumpy=True)._value
print(positions[dummy_atom_idx])

# import ipdb
# ipdb.set_trace()
simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH, SIM_TRAJ),
                                                                DCD_REPORT_STEPS))
                                                                #atomSubset=protein_ligand_idxs))

simulation.reporters.append(omma.CheckpointReporter(checkpoint_path,
                                                    CHECKPOINT_REPORTER_STEPS))

simulation.reporters.append(
    omma.StateDataReporter(
        LOG_FILE,
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

get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
omm_state = simulation.context.getState(**get_state_kwargs)
positions = omm_state.getPositions(asNumpy=True)._value
print(positions[dummy_atom_idx])