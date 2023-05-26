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

from metadynamics import *
import mdtraj as mdj
import parmed as pmd
import time

sys.path.append('../')
from  BFEE2_CV import EulerAngle_wall, Translation_restraint, Orientaion_restraint, RMSD_harmonic
from Euleranglesplugin import EuleranglesForce

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
        print('\n *************%64s : %16s : %8.3f ns/day ***************\n' % (description, platform_name, ns_per_day))
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
NUM_STEPS = 5000000 # 500000 = 1ns
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

inputs_dir = osp.realpath('../../openmm_plumed/inputs')
prmfile = osp.join(inputs_dir, 'complex.prmtop')
pdb_file = osp.join(inputs_dir, 'complex_bfee2.pdb')
pdb = mdj.load_pdb(pdb_file)
coords = omma.pdbfile.PDBFile(pdb_file).getPositions()

# protein and type!="H"'
protein_ligand_idxs = pdb.topology.select('protein or resname "MOL"')
ligand_idxs = pdb.topology.select('resname "MOL" and type!="H"')
protein_idxs = pdb.topology.select('protein and type!="H"')
ref_pos = omma.pdbfile.PDBFile(pdb_file).getPositions()


prmtop = omma.amberprmtopfile.AmberPrmtopFile(prmfile)
#coords = omma.amberinpcrdfile.AmberInpcrdFile(coodsfile).getPositions()
checkpoint_path = osp.join(OUTPUTS_PATH, CHECKPOINT)


# build the system
system = prmtop.createSystem(nonbondedMethod=omma.PME,
                            nonbondedCutoff=1*unit.nanometer,
                            constraints=omma.HBonds)

# atm, 300 K, with volume move attempts every 50 steps
barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)
system.addForce(barostat)
# Translation restraint on protein
dummy_atom_pos = omm.vec3.Vec3(4.27077094, 3.93215937, 3.84423549)*unit.nanometers 
translation_res = Translation_restraint(protein_idxs, dummy_atom_pos,
                                 force_const=41840*unit.kilojoule_per_mole/unit.nanometer**2) #41840
system.addForce(translation_res)
# Translation restraint on protein
dummy_atom_pos = omm.vec3.Vec3(4.27077094, 3.93215937, 3.84423549)*unit.nanometers 
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

eulertheta_harmonic_wall = EulerAngle_wall(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(),
                                           angle="Theta",
                                           lowerwall=-15.0, # fails when passing with units -15.0* unit.degree
                                           upperwall=15.0,
                                           force_const=100)#*unit.kilojoule_per_mole/unit.degree**2)
# import ipdb
# ipdb.set_trace()
eulertheta_harmonic_wall.setForceGroup(30)
system.addForce(eulertheta_harmonic_wall)
# #omma.metadynamics.BiasVariable using modefied version from biosimspace
sigma_eulerTheta = 1
eulertheta_cv = EuleranglesForce(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(), "Theta")
eulertheta_bias = BiasVariable(eulertheta_cv, minValue=-20.0, maxValue=20.0, 
                                           biasWidth=sigma_eulerTheta, periodic=False, gridWidth=400)

bias = 15.0
meta = Metadynamics(system, [eulertheta_bias], 
                    TEMPERATURE,
                    biasFactor=bias,
                    height=0.01*unit.kilojoules_per_mole,
                    frequency=1000,
                    saveFrequency=1000,
                    biasDir=".")
# make the integrator
report_timing(system, coords, "Run time for this step:")