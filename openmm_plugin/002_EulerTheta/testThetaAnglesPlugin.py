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
sys.path.append('../utils')
from  BFEE2_CV import EulerAngle_wall, Translation_restraint, Orientaion_restraint, RMSD_harmonic 
from Euleranglesplugin import EuleranglesForce


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
NUM_STEPS = 5000 #10000000 # 500000 = 1ns
REPORT_STEPS = 50

OUTPUTS_PATH = osp.realpath(f'outputs')
SIM_TRAJ = 'traj.dcd'

#
if not osp.exists(OUTPUTS_PATH):
    os.makedirs(OUTPUTS_PATH)
    
# the inputs directory and files we need
inputs_dir = osp.realpath(f'../../openmm_plumed/inputs')

prmfile = osp.join(inputs_dir, 'complex.prmtop')
prmtop = omma.amberprmtopfile.AmberPrmtopFile(prmfile)

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


# eulertheta_cv = EuleranglesForce(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(), "Theta") 
# eulertheta_cv.setForceGroup(29)
# system.addForce(eulertheta_cv)

eulertheta_harmonic_wall = EulerAngle_wall(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(),
                                           angle="Theta",
                                           lowerwall=-5.0, #*unit.degree,
                                           upperwall=5.0, #*unit.degree,
                                           force_const=1000)#*unit.kilojoule_per_mole/unit.degree**2)

eulertheta_harmonic_wall.setForceGroup(30)
system.addForce(eulertheta_harmonic_wall)

sigma_eulerTheta = 0.01
eulertheta_cv = EuleranglesForce(ref_pos, ligand_idxs.tolist(), protein_idxs.tolist(), "Theta")
eulertheta_bias = omma.metadynamics.BiasVariable(eulertheta_cv, minValue=-20.0, maxValue=20.0, 
                                           biasWidth=sigma_eulerTheta, periodic=False, gridWidth=400)

bias = 15.0
meta = omma.metadynamics.Metadynamics(system, [eulertheta_bias], 
                    TEMPERATURE,
                    biasFactor=bias,
                    height=0.5*unit.kilojoules_per_mole,
                    frequency=1000,
                    saveFrequency=1000,
                    biasDir=".")

integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION)

# for i, f in enumerate(system.getForces()):
#     f.setForceGroup(i)
simulation = omma.Simulation(prmtop.topology, system, integrator, platform)
simulation.context.setPositions(ref_pos)
simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH, SIM_TRAJ),
                                                                REPORT_STEPS,
                                                                atomSubset=protein_ligand_idxs))


#forces_file = open("biases.dat", 'w')
data = []
forces = []
for x in range(0, int(NUM_STEPS/REPORT_STEPS)):
    simulation.step(REPORT_STEPS)
    # state = simulation.context.getState(getEnergy=True, getForces=True, groups={29})
    # eulertheta = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    # data.append(eulertheta)
    state = simulation.context.getState(getEnergy=True, getForces=True, groups={30})
    wall_energy= state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    #wall_forces = state.getForces(asNumpy=True)
    wall_cv = eulertheta_harmonic_wall.getCollectiveVariableValues(simulation.context)
    #print(f"Euler Theta= {eulertheta}, wall_cv={wall_cv[0]}, wall_energy={wall_energy}\n ")
    calculated_energy = 0.5 * 1000 * (min(wall_cv[0] - (-5), 0)**2 + max(wall_cv[0]-5, 0)**2)
    print(f"wall_cv={wall_cv[0]}, calc_energy={calculated_energy}, wall_energy={wall_energy}")
np.save('COLVAR', np.array(data))
np.save('forces', np.array(forces))
