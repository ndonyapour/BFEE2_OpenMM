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



sys.path.append('../utils')
from BFEE2_CV import *
from Euleranglesplugin import EuleranglesForce
from Polaranglesplugin import PolaranglesForce
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
REPORTER_STEPS = 1000
DCD_REPORTER_STEPSS = 5000
HILLS_REPORTER_STEPS = 1000
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
# 4.2773213,3.9284525,3.8494892
dummy_atom_pos = omm.vec3.Vec3(*com[0])*unit.nanometers
translation_res = Translation_restraint(protein_idxs, dummy_atom_pos,
                                 force_const=41840*unit.kilojoule_per_mole/unit.nanometer**2) #41840
system.addForce(translation_res)

# Orientaion restraint
q_centers = [1.0, 0.0, 0.0, 0.0]
q_force_consts = [8368*unit.kilojoule_per_mole/unit.nanometer**2 for _ in range(4)]
orientaion_res = Orientaion_restraint(ref_pos, protein_idxs.tolist(), q_centers, q_force_consts)
system.addForce(orientaion_res)

# RMSD CV
rmsd_cv = omm.RMSDForce(ref_pos, ligand_idxs)
rmsd_cv.setForceGroup(20)
system.addForce(rmsd_cv)

# Euler Theta
eulertheta_cv = EuleranglesForce(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(), "Theta")
eulertheta_cv.setForceGroup(21)
system.addForce(eulertheta_cv)

# Euler Phi
eulerphi_cv = EuleranglesForce(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(), "Phi")
eulerphi_cv.setForceGroup(22)
system.addForce(eulerphi_cv)

# Euler Phi
eulerpsi_cv = EuleranglesForce(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(), "Psi")
eulerpsi_cv.setForceGroup(23)
system.addForce(eulerpsi_cv)

# Polar Theta
polartheta_cv = PolaranglesForce(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(), "Theta")
polartheta_cv.setForceGroup(24)
system.addForce(polartheta_cv)

# Polar Phi
polarphi_cv = PolaranglesForce(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(), "Phi")
polarphi_cv.setForceGroup(25)
system.addForce(polarphi_cv)

integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION) 

simulation = omma.Simulation(prmtop.topology, system, integrator, platform, prop)
simulation.context.setPositions(ref_pos)



simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH, SIM_TRAJ),
                                                                DCD_REPORTER_STEPSS,
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

print("Start simulations from the crystal structures")
CVs = []
file = open("COLVAR", "w")
for x in range(0, int(NUM_STEPS/COLVAR_REPORTER_STEPS)):
    simulation.step(COLVAR_REPORTER_STEPS)
    current_CVs = []
    for f_id in [20, 21, 22, 23, 24, 25]:
        state = simulation.context.getState(getEnergy=True, getForces=True, groups={f_id})
        current_CVs.append(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    CVs.append(current_CVs)
    line = '\t'.join([str(n) for n in current_CVs])
    line += "\n"
    file.write(line)
    file.flush()

np.save('COLVAR', np.array(CVs))