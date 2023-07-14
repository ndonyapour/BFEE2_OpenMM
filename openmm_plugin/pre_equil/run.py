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
DEVICEINDEX = '2,3,4,5,6'
MIN_NUM_STEPS = 10000
NVT_NUM_STEPS = 100000  #200ps
NPT_NUM_STEPS = 5000000 # 1ns 

OUTPUTS_PATH = osp.realpath(f'outputs_largebox')

if not osp.exists(OUTPUTS_PATH):
    os.makedirs(OUTPUTS_PATH)
    
# the inputs directory and files we need
inputs_dir = osp.realpath(f'../setup_inputs/large_box')
prmfile = osp.join(inputs_dir, 'complex_ions.prmtop')
coodsfile = osp.join(inputs_dir, 'complex_ions.rst7')

prmtop = pmd.load_file(prmfile, coodsfile)

platform = omm.Platform.getPlatformByName(PLATFORM)
properties = dict(Precision=PRECISION, DeviceIndex=DEVICEINDEX)


# minimize the structure
min_save_path = osp.join(OUTPUTS_PATH, 'complex_largebox_min.rst7')
min_coords = minimize(prmtop, platform, properties, prmtop.positions, 
                     num_steps=MIN_NUM_STEPS, 
                     save_path=min_save_path)

# NVT EQUIL
nvt_save_path = osp.join(OUTPUTS_PATH, 'complex_largebox_nvt.rst7')
nvt_coords = equil_NVT(prmtop, platform, properties, min_coords, 
                      num_steps=NVT_NUM_STEPS, 
                      save_path=nvt_save_path)

# NPT EQUIL
npt_save_path = osp.join(OUTPUTS_PATH, 'complex_largebox_npt.rst7')
npt_coords = equil_NPT(prmtop, platform, properties, nvt_coords, 
                      num_steps=NPT_NUM_STEPS ,
                      save_path=npt_save_path)


pdb_save_path = osp.join(OUTPUTS_PATH, 'complex_npt.pdb')
prmtop.coordinates = npt_coords
prmtop.save(pdb_save_path, format='pdb', overwrite=True)