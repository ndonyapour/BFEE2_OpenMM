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
sys.path.append('../')
from  BFEE2_CV import RMSD_CV, Translation_CV

#omm.Platform.loadPluginsFromDirectory("/home/ndonyapour/miniconda3/pkgs/openmm-torch-0.8-cuda112py39h83a068c_2/lib/plugins")

DEVICEINDEX = '2,5'
NUM_STEPS = 5000
PLATFORM = 'CUDA'
PRECISION = 'mixed'
TEMPERATURE = 300.0 * unit.kelvin
FRICTION_COEFFICIENT = 1.0 / unit.picosecond
STEP_SIZE = 0.002 * unit.picoseconds
PRESSURE = 1.0 * unit.atmosphere
VOLUME_MOVE_FREQ = 50


def report_timing(system, positions, description):
    """Report timing on all available platforms."""
    timestep = 2.0*unit.femtoseconds
    for platform_name in ['CUDA']:
        platform = omm.Platform.getPlatformByName(platform_name)

        integrator = omm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, timestep)
        properties = dict(Precision=PRECISION, DeviceIndex=DEVICEINDEX)
        context = omm.Context(system, integrator, platform, properties)
        context.setPositions(positions)
        # Warm up the integrator
        integrator.step(10)
        # Time integration
        initial_time = time.time()
        integrator.step(NUM_STEPS)
        final_time = time.time()
        elapsed_time = (final_time - initial_time) * unit.seconds
        ns_per_day = NUM_STEPS * timestep / elapsed_time / (unit.nanoseconds / unit.day)
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

inputs_dir = osp.realpath(f'../../openmm_plumed/inputs')


prmfile = osp.join(inputs_dir, 'complex.prmtop')
prmtop = omma.amberprmtopfile.AmberPrmtopFile(prmfile)



pdb_file = osp.join(inputs_dir, 'complex_bfee2.pdb')
pdb = mdj.load_pdb(pdb_file)
coords = omma.pdbfile.PDBFile(pdb_file).getPositions()

prmtop = omma.amberprmtopfile.AmberPrmtopFile(prmfile)
#coords = omma.amberinpcrdfile.AmberInpcrdFile(coodsfile).getPositions()



# build the system
system = prmtop.createSystem(nonbondedMethod=omma.PME,
                            nonbondedCutoff=1*unit.nanometer,
                            constraints=omma.HBonds)

# atm, 300 K, with volume move attempts every 50 steps
barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)
system.addForce(barostat)


protein_ligand_idxs = pdb.topology.select('protein or resname "MOL"')
ligand_idxs = pdb.topology.select('resname "MOL" and type!="H"')
protein_idxs = pdb.topology.select('protein and type!="H"')
ligand_ref_file = osp.join(inputs_dir, 'complex_bfee2.pdb')
ligand_ref_pos = omma.pdbfile.PDBFile(ligand_ref_file).getPositions()

# add Pytorch force 
# quaternion_force = TorchForce('inputs/quaternion.pt')
# system.addForce(quaternion_force)
# quaternion_force.setOutputsForces(True)

# quaternion_force = TorchForce('inputs/quaternion.pt')
# system.addForce(quaternion_force)
# quaternion_force.setOutputsForces(True)


# RMSD 

for _ in range(7):
    cv = omm.RMSDForce(ligand_ref_pos, ligand_idxs)


    system.addForce(cv)


# rmsd_cv, rmsd_harmonic_wall = RMSD_CV(ligand_ref_pos, ligand_idxs,
#                                     lowerwall=0.0*unit.nanometer,
#                                     upperwall=0.3*unit.nanometer,
#                                     force_const=2000*unit.kilojoule_per_mole/unit.nanometer**2)


# system.addForce(rmsd_harmonic_wall)

# Translation CV
dummy_atom_pos = omm.vec3.Vec3(4.27077094, 3.93215937, 3.84423549)*unit.nanometers
translation_res = Translation_CV(protein_idxs, dummy_atom_pos,
                                 force_const=41840*unit.kilojoule_per_mole/unit.nanometer**2)
system.addForce(translation_res)


rmsd_cv = omm.RMSDForce(ligand_ref_pos, ligand_idxs)
# wall_energy_exp = "0.5*k*(min(0,RMSD-lowerwall)^2+max(0,RMSD-upperwall)^2)"
# wall_restraint_force = omm.CustomCVForce(wall_energy_exp)
# wall_restraint_force.addCollectiveVariable('RMSD',  rmsd_cv)
# wall_restraint_force.addGlobalParameter('lowerwall', 0)
# wall_restraint_force.addGlobalParameter('upperwall', 0.3)
# wall_restraint_force.addGlobalParameter("k", 1000)

rmsd_bias = omma.metadynamics.BiasVariable(rmsd_cv, 0.0, 5.0, 0.05, False, gridWidth=100)
meta = omma.metadynamics.Metadynamics(system, [rmsd_bias], 
                                        TEMPERATURE,
                                        biasFactor=10,
                                        height=1.5 * unit.kilojoules_per_mole,
                                        500,
                                        biasDir=".",
                                        saveFrequency=1000)

# make the integrator
report_timing(system, coords, "Run time for this step:")
