import os.path as osp

import numpy as np
import mdtraj as mdj 

import torch
from TorchBFEE2.utils.utils import save_quaternion
from TorchBFEE2.models.quaternion import Quaternion


inputs_dir = osp.realpath(f'../../openmm_plumed/NAMD_test')
refpdb_path = osp.join(inputs_dir, 'complex.pdb')
pdb = mdj.load_pdb(refpdb_path)

protein_idxs = pdb.topology.select('protein and type!="H"')
ligand_idxs = pdb.topology.select('resname "MOL" and type!="H"')


model_save_path = "inputs/quaternion.pt"
model = Quaternion(ligand_idxs.tolist(), pdb.xyz[0, ligand_idxs].tolist())
#pos = pdb.xyz[0, 0:2].tolist()
#model = Qrotation([1, 2, 3], pos)
script_module = torch.jit.script(model)
script_module = torch.jit.freeze(script_module.eval())
script_module.save(model_save_path)
model = torch.jit.load(model_save_path)

# # import ipdb
# # ipdb.set_trace()
# script_module = torch.jit.freeze(script_module.eval())
# script_module.save(model_save_path)
# save_quaternion(protein_idxs.tolist(), pdb.xyz[:, protein_idxs].tolist(), save_path=model_save_path)
#model = torch.jit.load(model_save_path)

# # load the dcd file 
eq_dcd_path = osp.join(inputs_dir, 'eq.dcd')
dcd = mdj.load_dcd(eq_dcd_path, pdb)

qs = []
for coords in dcd.xyz[:5]:
    print(coords.shape)
    coords = torch.from_numpy(coords)
    coords.requires_grad = True
    q = model.forward(coords)[0]
    q.backward()
    # print(coords.grad().shape())
    # qs.append(q.detach().cpu().numpy())

np.save("inputs/eq_quaternion.npy", np.array(qs))