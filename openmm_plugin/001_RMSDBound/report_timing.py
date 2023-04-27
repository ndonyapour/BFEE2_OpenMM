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

sys.path.append('../')
from  BFEE2_CV import RMSD_wall, Translation_restraint, Orientaion_restraint

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
# for openmm simulations (0,0,0) is the left corner of box, cerntering molceules to (0, 0, 0) cuases proplems.
#coodsfile = osp.join(inputs_dir, 'complex.rst7')
pdb_file = osp.join(inputs_dir, 'complex_bfee2.pdb')
pdb = mdj.load_pdb(pdb_file)
coords = omma.pdbfile.PDBFile(pdb_file).getPositions()


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

# protein and type!="H"'
protein_ligand_idxs = pdb.topology.select('protein or resname "MOL"')
ligand_idxs = pdb.topology.select('resname "MOL" and type!="H"')
protein_idxs = pdb.topology.select('protein and type!="H"')
ref_pos = omma.pdbfile.PDBFile(pdb_file).getPositions()

# Define walls on the RMSD CV
rmsd_harmonic_wall = RMSD_wall(ref_pos, ligand_idxs,
                            lowerwall=0.0*unit.nanometer,
                            upperwall=0.3*unit.nanometer,
                            force_const=2000*unit.kilojoule_per_mole/unit.nanometer**2)


system.addForce(rmsd_harmonic_wall)

# run metadynamics on rmsd
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
for i in range(0):
    q_restraint = Orientaion_restraint(ref_pos, protein_idxs.tolist(), i, center=q_centers[0]*unit.nanometer)
    system.addForce(q_restraint)

# make the integrator
report_timing(system, coords, "Run time for this step:")