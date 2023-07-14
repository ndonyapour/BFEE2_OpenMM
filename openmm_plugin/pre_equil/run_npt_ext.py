import sys
import os
import os.path as osp
import pickle
import random

import numpy as np
import pickle as pkl

import openmm.app as omma
import openmm as omm
from openmmplumed import PlumedForce
import simtk.unit as unit


import mdtraj as mdj
import parmed as pmd
import time
from utils import minimize, equil_NVT, equil_NPT


# Platform used for OpenMM which uses different hardware computation
# kernels. Options are: Reference, CPU, OpenCL, CUDA.

PLATFORM = 'CUDA'
PRECISION = 'mixed'
DEVICEINDEX = '2,3,4,5,6,7'
EQUIL_NPT_NUM_STEPS = 25000000  #10ns

OUTPUTS_PATH = osp.realpath(f'outputs_ext2')

if not osp.exists(OUTPUTS_PATH):
    os.makedirs(OUTPUTS_PATH)
    
# the inputs directory and files we need
inputs_dir = osp.realpath(f'../setup_inputs')
prmfile = osp.join(inputs_dir, 'complex_ions.prmtop')
coodsfile = osp.join('outputs', 'complex_nvt.rst7')

prmtop = pmd.load_file(prmfile, coodsfile)

platform = omm.Platform.getPlatformByName(PLATFORM)
properties = dict(Precision=PRECISION, DeviceIndex=DEVICEINDEX)


npt_save_path = osp.join(OUTPUTS_PATH, 'complex_npt.rst7')
npt_coords = equil_NPT(prmtop, platform, properties, prmtop.positions, 
                      num_steps=EQUIL_NPT_NUM_STEPS ,
                      save_path=npt_save_path)


pdb_save_path = osp.join(OUTPUTS_PATH, 'complex_npt.pdb')
prmtop.coordinates = npt_coords
prmtop.save(pdb_save_path, format='pdb', overwrite=True)