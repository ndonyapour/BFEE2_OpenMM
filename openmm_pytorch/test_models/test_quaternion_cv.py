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
from openmmtorch import TorchForce

#from BFEE2_CV import RMSD_CV, Translation_CV
# sys.path.append('../')
# from  BFEE2_CV import RMSD_CV, Translation_CV
# install openmm-torch and get path by seraching the name 
omm.Platform.loadPluginsFromDirectory("/home/ndonyapour/miniconda3/pkgs/openmm-torch-0.8-cuda112py39h83a068c_2/lib/plugins")
    # #'/home/ndonyapour/miniconda3/envs/torchmodels/lib/plugins')
# '/home/ndonyapour/miniconda3/envs/ommtest/lib/plugins')  # got from openmm import *, print(version.openmm_library_path)

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
LOG_REPORTER_STEPS = 500
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

pdb = mdj.load_pdb(osp.join(inputs_dir, 'complex_bfee2.pdb'))
coords = omma.pdbfile.PDBFile(osp.join(inputs_dir, 'complex_bfee2.pdb')).getPositions()

# protein and type!="H"'
protein_ligand_idxs = pdb.topology.select('protein or resname "MOL"')
ligand_idxs = pdb.topology.select('resname "MOL" and type!="H"')
protein_idxs = pdb.topology.select('protein and type!="H"')
ligand_ref_file = osp.join(inputs_dir, 'ligand.pdb')
ligand_ref_pos = omma.pdbfile.PDBFile(ligand_ref_file).getPositions()
# add disulfide bonds to the topology

# add a dummy atom (ref protein COM) and set its mass to zero


system = prmtop.createSystem(nonbondedMethod=omma.PME,
                            nonbondedCutoff=1*unit.nanometer,
                            constraints=omma.HBonds)

barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)
# # add it as a "Force" to the system
system.addForce(barostat)

# Define CVs and restraint forces

# RMSD CV
# rmsd_cv, rmsd_harmonic_wall = RMSD_CV(ligand_ref_pos, ligand_idxs,
#                                       lowerwall=0.0*unit.nanometer,
#                                       upperwall=0.3*unit.nanometer,
#                                       force_const=2000*unit.kilojoule_per_mole/unit.nanometer**2)


# system.addForce(rmsd_harmonic_wall)

# # Translation CV
# dummy_atom_pos = omm.vec3.Vec3(4.27077094, 3.93215937, 3.84423549)*unit.nanometers
# translation_res = Translation_CV(protein_idxs, dummy_atom_pos,
#                                  force_const=41840*unit.kilojoule_per_mole/unit.nanometer**2)
#system.addForce(translation_res)

# add the quaternion 
quaternion_force = TorchForce('inputs/quaternion.pt')


system.addForce(quaternion_force)

#ext = BiasVariable(extent, 0.0, 0.95, sigma_ext, False, gridWidth=200)

# make the integrator
integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION)

simulation = omma.Simulation(prmtop.topology, system, integrator, platform, prop)
# if osp.exists(STAR_CHECKPOINT):
#     print("Start from checkpoint")
#     simulation.loadCheckpoint(STAR_CHECKPOINT)
# else:
#     print("Can not find the checkpoint")
simulation.context.setPositions(coords)

simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH, SIM_TRAJ),
                                                                DCD_REPORT_STEPS,
                                                                atomSubset=protein_ligand_idxs))



simulation.reporters.append(omma.CheckpointReporter(checkpoint_path,
                                                    CHECKPOINT_REPORTER_STEPS))

simulation.reporters.append(
    omma.StateDataReporter(
        sys.stdout,
        20,
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