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
from Euleranglesplugin import EuleranglesForce
from reporters import HILLSReporter, COLVARReporter

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
NUM_STEPS = 5000000 # 500000 = 1ns   #5000000
DCD_REPORTER_STEPS = 5000
COLVAR_REPORTER_STEPS = 5000
HILLS_REPORTER_STEPS = 1000
CHECKPOINT_REPORTER_STEPS =  5000
LOG_REPORTER_STEPS = 5000
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
com = mdj.compute_center_of_mass(pdb, select='resname "MOL" and type!="H"')
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

# fails when passing with CV units of unit.degree
eulertheta_harmonic_wall = EulerAngle_wall(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(),
                                           angle="Theta",
                                           lowerwall=-15.0,
                                           upperwall=15.0,
                                           force_const=100)

system.addForce(eulertheta_harmonic_wall)

sigma = 0.6
eulertheta_cv = EuleranglesForce(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(), "Theta")
eulertheta_bias = omma.metadynamics.BiasVariable(eulertheta_cv, minValue=-20.0, maxValue=20.0,
                                                 biasWidth=sigma, periodic=False, gridWidth=400)

bias = 15.0
meta = omma.metadynamics.Metadynamics(system, [eulertheta_bias],
                    TEMPERATURE,
                    biasFactor=bias,
                    height=0.01*unit.kilojoules_per_mole,
                    frequency=HILLS_REPORTER_STEPS)
                    # saveFrequency=HILLS_REPORTER_STEPS,
                    # biasDir=".")

integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION)


simulation = omma.Simulation(prmtop.topology, system, integrator, platform, prop)
if osp.exists(STAR_CHECKPOINT):
    print(f"Start Simulation from checkpoint {STAR_CHECKPOINT}")
    simulation.loadCheckpoint(STAR_CHECKPOINT)
else:
    print("Can not find the checkpoint")


simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH, SIM_TRAJ),
                                                                DCD_REPORTER_STEPS,
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

simulation.reporters.append(HILLSReporter(meta,
                                          "./",
                                          sigma,
                                          reportInterval=HILLS_REPORTER_STEPS,
                                          cvname="eulerTheta"))
simulation.reporters.append(COLVARReporter(meta, './',
                                           [rmsd_res, orientaion_res],
                                           reportInterval=COLVAR_REPORTER_STEPS))

start_time = time.time()
meta.step(simulation, NUM_STEPS)
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