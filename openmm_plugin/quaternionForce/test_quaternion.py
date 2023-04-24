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

sys.path.append('../../openmm_pytorch')
from BFEE2_CV import RMSD_CV, Translation_CV
from Quaternionplugin import QuaternionForce

DEVICEINDEX = '2'
PRECISION = 'mixed'


#omm.Platform.loadPluginsFromDirectory("/home/ndonyapour/miniconda3/pkgs/openmm-torch-0.8-cuda112py39h83a068c_2/lib/plugins")

def getCV(system, positions, description, groups):
    """Report timing on all available platforms."""
    timestep = 2.0*unit.femtoseconds
    nsteps = 10
    for platform_name in ['CPU']:
        platform = omm.Platform.getPlatformByName(platform_name)

        integrator = omm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, timestep)
        properties = dict(Precision=PRECISION, DeviceIndex=DEVICEINDEX)
        context = omm.Context(system, integrator, platform)#, properties)
        context.setPositions(positions)
        # Warm up the integrator
        integrator.step(10)
        # Time integration
        initial_time = time.time()
        for i in range(10):
            integrator.step(nsteps)
            state = context.getState(getForces=True, getEnergy=True,
                                     getPositions=True, getVelocities=True,
                                     groups={groups})
            q = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            print(f"{nsteps*i}\t{q}")
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

inputs_dir = osp.realpath(f'../../openmm_plumed/inputs')
prmfile = osp.join(inputs_dir, 'complex.prmtop')
# for openmm simulations (0,0,0) is the left corner of box, cerntering molceules to (0, 0, 0) cuases proplems.
#coodsfile = osp.join(inputs_dir, 'complex.rst7')
plumed_file = osp.realpath('plumed.dat')
pdb_file = osp.join(inputs_dir, 'complex_bfee2.pdb')
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

ligand_idxs = mdj.load_pdb(pdb_file).topology.select('resname "MOL" and type!="H"')
protein_idxs = mdj.load_pdb(pdb_file).topology.select('protein and type!="H"')

# add RMSD Pluging Force
qforce = QuaternionForce(coords, protein_idxs.tolist())
qforce_qroup = 30
qforce.setForceGroup(qforce_qroup)
system.addForce(qforce)

# wall_energy_exp = "0.5*k*(min(0,RMSD-lowerwall)^2+max(0,RMSD-upperwall)^2)"
# wall_restraint_force = omm.CustomCVForce(wall_energy_exp)
# wall_restraint_force.addCollectiveVariable('RMSD',  rmsd_plugin)
# wall_restraint_force.addGlobalParameter('lowerwall', 0.0*unit.nanometer)
# wall_restraint_force.addGlobalParameter('upperwall', 0.3*unit.nanometer)
# wall_restraint_force.addGlobalParameter("k", 2000*unit.kilojoule_per_mole/unit.nanometer**2)
# system.addForce(wall_restraint_force)


# make the integrator
getCV(system, coords, "Run time for this step:", qforce_qroup)
