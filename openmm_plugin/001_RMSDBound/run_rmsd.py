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

from metadynamics import *


#from BFEE2_CV import RMSD_CV, Translation_CV
sys.path.append('../')
from  BFEE2_CV import RMSD_wall, Translation_restraint, Orientaion_restraint


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
DCD_REPORT_STEPS = 100
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
system.addForce(rmsd_harmonic_wall)
# run metadynamics on rmsd

rmsd_cv = omm.RMSDForce(ref_pos, ligand_idxs)

#omma.metadynamics.BiasVariable using modefied version from biosimspace
sigma_rmsd = 0.01
rmsd_bias = BiasVariable(rmsd_cv, minValue=0.0*unit.nanometer, maxValue=0.3*unit.nanometer, 
                                           biasWidth=sigma_rmsd*unit.nanometer, periodic=False, gridWidth=100)

bias = 10
meta = Metadynamics(system, [rmsd_bias], 
                                        TEMPERATURE,
                                        biasFactor=bias,
                                        height=1*unit.kilojoules_per_mole,
                                        frequency=1000,
                                        saveFrequency=5000,
                                        biasDir=".")
# Translation restraint on protein
dummy_atom_pos = omm.vec3.Vec3(4.27077094, 3.93215937, 3.84423549)*unit.nanometers 
translation_res = Translation_restraint(protein_idxs, dummy_atom_pos,
                                 force_const=41840*unit.kilojoule_per_mole/unit.nanometer**2) #41840
system.addForce(translation_res)

# Orientaion restraint
q_centers = [1, 0, 0, 0]
q_cvs = []
for i in range(4):
    q_restraint = Orientaion_restraint(ref_pos, protein_idxs.tolist(), i, 
                                       center=q_centers[0]*unit.nanometer, 
                                       force_const=8368*unit.kilojoule_per_mole/unit.nanometer**2) # 8368
    system.addForce(q_restraint)
    q_cvs.append(q_restraint)
    
integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION)

for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)

simulation = omma.Simulation(prmtop.topology, system, integrator, platform, prop)
if osp.exists(STAR_CHECKPOINT):
    print(f"Start Simulation from checkpoint {STAR_CHECKPOINT}")
    simulation.loadCheckpoint(STAR_CHECKPOINT)
else:
    print("Can not find the checkpoint")
#simulation.context.setPositions(ref_pos)

simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH, SIM_TRAJ),
                                                                DCD_REPORT_STEPS,
                                                                atomSubset=protein_ligand_idxs))



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

# Create PLUMED compatible HILLS file.
file = open('HILLS','w')
file.write('#! FIELDS time rmsd sigma_rmsd height biasf\n')
file.write('#! SET multivariate false\n')
file.write('#! SET kerneltype gaussian\n')

# Initialise the collective variable array.
current_cvs = list(meta.getCollectiveVariables(simulation))
for idx, q in enumerate(q_cvs):
    # print(idx, q.getCollectiveVariableValues(simulation.context)[0], " ")
    current_cvs.append(q.getCollectiveVariableValues(simulation.context)[0])
    
hill_hight = meta.getHillHeight(simulation)
colvar_array = np.array([current_cvs])

rtime = 0
write_line = f'{rtime:15} {colvar_array[0][0]:20.16f}          {sigma_rmsd} {meta.getHillHeight(simulation):20.16f}          {bias}\n'
file.write(write_line)

# Run the simulation
report_step = 5000
start_time = time.time()
#force_file = open("forces.txt", 'w')
for x in range(0, int(NUM_STEPS/report_step)):
    meta.step(simulation, report_step)
    current_cvs = list(meta.getCollectiveVariables(simulation))
    for idx, q in enumerate(q_cvs):
        current_cvs.append(q.getCollectiveVariableValues(simulation.context)[0])


    #data = []
    # for i, f in enumerate(system.getForces()):
    #     state = simulation.context.getState(getEnergy=True, groups={i})
    #     data.append(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    #     #print(f.getName(), state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        
    #force_file.write("\t".join(str(round(item, 3)) for item in data)+"\n")
    wall_rmsd = rmsd_harmonic_wall.getCollectiveVariableValues(simulation.context)[0]
    #print(f"{wall_rmsd} \t {current_cvs[0]}")
    #print(f"{current_cvs[0]}")
    rtime = int((x+1) * 0.002*report_step)
    write_line = f'{rtime:15} {current_cvs[0]:20.16f}          {sigma_rmsd} {meta.getHillHeight(simulation):20.16f}          {bias}\n'
    colvar_array = np.append(colvar_array, [current_cvs], axis=0)
    np.save('COLVAR.npy', colvar_array)
    line = colvar_array[x+1]
   

    file.write(write_line)
    file.flush()

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