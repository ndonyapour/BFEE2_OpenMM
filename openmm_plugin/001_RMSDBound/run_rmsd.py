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


#from BFEE2_CV import RMSD_CV, Translation_CV
sys.path.append('../')
from  BFEE2_CV import RMSD_wall, Translation_restraint, Orientaion_restraint
from Quaternionplugin import QuaternionForce

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

# Define CVs and restraint forces

# RMSD CV

rmsd_harmonic_wall = RMSD_wall(ref_pos, ligand_idxs,
                            lowerwall=0.0*unit.nanometer,
                            upperwall=0.3*unit.nanometer,
                            force_const=2000*unit.kilojoule_per_mole/unit.nanometer**2)

# run metadynamics on rmsd

system.add(rmsd_harmonic_wall)

rmsd_cv = omm.RMSDForce(ref_pos, ligand_idxs)
rmsd_bias = omma.metadynamics.BiasVariable(rmsd_cv, minValue=0.0*unit.nanometer, maxValue=0.3*unit.nanometer, 
                                           biasWidth=0.02*unit.nanometer, periodic=False, gridWidth=100)

meta = omma.metadynamics.Metadynamics(system, [rmsd_bias], 
                                        TEMPERATURE,
                                        biasFactor=20,
                                        height=2*unit.kilojoules_per_mole,
                                        frequency=500,
                                        saveFrequency=5000,
                                        biasDir=".")
# Translation restraint on protein
dummy_atom_pos = omm.vec3.Vec3(4.27077094, 3.93215937, 3.84423549)*unit.nanometers
translation_res = Translation_restraint(protein_idxs, dummy_atom_pos,
                                 force_const=41840*unit.kilojoule_per_mole/unit.nanometer**2)
system.addForce(translation_res)

#add the quaternion 
q_centers = [1, 0, 0, 0]
for i in range(4):
    q_restraint = Orientaion_restraint(ref_pos, protein_idxs.tolist(), i, center=q_centers[0]*unit.nanometer)
    system.addForce(q_restraint)

# make the integrator
integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION)

simulation = omma.Simulation(prmtop.topology, system, integrator, platform, prop)
if osp.exists(STAR_CHECKPOINT):
    print("Start from checkpoint")
    simulation.loadCheckpoint(STAR_CHECKPOINT)
else:
    print("Can not find the checkpoint")
#simulation.context.setPositions(ref_pos)

simulation.step(20)
print("Done")

# simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH, SIM_TRAJ),
#                                                                 DCD_REPORT_STEPS,
#                                                                 atomSubset=protein_ligand_idxs))



# simulation.reporters.append(omma.CheckpointReporter(checkpoint_path,
#                                                     CHECKPOINT_REPORTER_STEPS))

# simulation.reporters.append(
#     omma.StateDataReporter(
#         LOG_FILE,
#         LOG_REPORTER_STEPS,
#         step=True,
#         time=True,
#         potentialEnergy=True,
#         kineticEnergy=True,
#         totalEnergy=True,
#         volume=True,
#         temperature=True,
#         totalSteps=True,
#         separator=" ",
#     )
# )
# print("Start Simulation")
# start_time = time.time()
# simulation.step(NUM_STEPS)
# end_time = time.time()
# print("End Simulation")
# print(f"Run time = {np.round(end_time - start_time, 3)}s")
# simulation_time = round((STEP_SIZE * NUM_STEPS).value_in_unit(unit.nanoseconds),
#                            2)
# print(f"Simulation time: {simulation_time}ns")
# simulation.saveCheckpoint(osp.join(OUTPUTS_PATH, CHECKPOINT_LAST))

# # save final state and system
# get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
# omm_state = simulation.context.getState(**get_state_kwargs)
# # save the pkl files to the inputs dir
# with open(osp.join(OUTPUTS_PATH, SYSTEM_FILE), 'wb') as wfile:
#     pkl.dump(system, wfile)

# with open(osp.join(OUTPUTS_PATH, OMM_STATE_FILE), 'wb') as wfile:
#     pkl.dump(omm_state, wfile)
# print('Done making pkls. Check inputs dir for them!')