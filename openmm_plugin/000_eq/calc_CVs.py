import sys
import os.path as osp
import pandas as pd
import numpy as np

import mdtraj as mdj 

sys.path.append('../utils')
from geometric_CVs import GeometricCVs


inputs_dir = osp.realpath(f'../../openmm_plumed/inputs')


pdb_file = osp.join(inputs_dir, 'protein_ligand.pdb')
pdb = mdj.load_pdb(pdb_file)

protein_ligand_idxs = pdb.topology.select('protein or resname "MOL"')
ligand_idxs = pdb.topology.select('resname "MOL" and type!="H"')
protein_idxs = pdb.topology.select('protein and type!="H"')

ref_pos_pdb = mdj.load_pdb(osp.join(inputs_dir, 'complex_bfee2.pdb'))
ref_pos = ref_pos_pdb.xyz[0, protein_ligand_idxs]


# read the trajectory file 
traj = mdj.load_dcd('./outputs/traj.dcd', osp.join(inputs_dir,'protein_ligand.pdb'))
print(f"The number of frames: {traj.n_frames}")

# calculate CVs
gcv = GeometricCVs()
CVs = []
# rmsd
print("Calculating the RMSD CV")
rmsd = gcv.rmsd(pdb, traj, ligand_idxs)
CVs.append(rmsd)

# Euler angles 
print("Calculating Euler angles")
euler_angles = gcv.EuelrAngle(pdb.xyz[0], traj.xyz, ligand_idxs, protein_idxs, angle='all')
[CVs.append(l) for l in euler_angles.T]

# Polar angles
print("Calculating Polar angles")
polar_angles = gcv.PolarAngle(pdb.xyz[0], traj.xyz, ligand_idxs, protein_idxs, angle='all')
[CVs.append(l) for l in polar_angles.T]

# r 
print("Calculating the r CV")
r = gcv.r(traj, 'protein and type!="H"', 'resname "MOL" and type!="H"')
CVs.append(r)

# translation 
print("Calculating the translation CV")
translation = gcv.translation(pdb, traj, 'protein and type!="H"')
CVs.append(translation)

# orientation
print("Calculating the orientaionm CV")
orientaion = gcv.orientaion(pdb.xyz[0], traj.xyz, protein_idxs)
[CVs.append(l) for l in orientaion.T]

# save in a csv file
CV_names = ['rmsd', 'eulerTheta', 'eulerPhi', 'eulerPsi', 'polarTheta', 'polarPhi', 'r', 'translation', 'rot_q.w', 'rot_q.x', 'rot_q.y', 'rot_q.z'] 
data = {f'{CV_names[idx]}': CVs[idx] for idx in range(len(CVs))}
colvar = pd.DataFrame(data)
colvar.to_csv('COLVAR', sep=',')

