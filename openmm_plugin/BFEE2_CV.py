import openmm.app as omma
import openmm as omm

import simtk.unit as unit
from Quaternionplugin import QuaternionForce

def RMSD_wall(ref_pos, atom_idxs, lowerwall=0.0, upperwall=0.3, force_const=unit.kilojoule_per_mole/unit.nanometer**2):

    rmsd_cv = omm.RMSDForce(ref_pos, atom_idxs)
    # apply restraint if rmsd < lowerwall or rmsd > upperwall
    wall_energy_exp = "0.5*krmsd*(min(0,RMSD-lowerwall)^2+max(0,RMSD-upperwall)^2)"
    wall_restraint_force = omm.CustomCVForce(wall_energy_exp)
    wall_restraint_force.addCollectiveVariable("RMSD",  rmsd_cv)
    wall_restraint_force.addGlobalParameter("lowerwall", lowerwall)
    wall_restraint_force.addGlobalParameter("upperwall", upperwall)
    wall_restraint_force.addGlobalParameter("krmsd", force_const)

    return wall_restraint_force

def Translation_restraint(atom_idxs, dummy_atom_pos, force_const=41840*unit.kilojoule_per_mole/unit.nanometer**2):

    # 1/2 * k * distance(com-dummy_atom)^2
    translation_restraint = omm.CustomCentroidBondForce(1, "0.5*kt*((x1-dx)^2+(y1-dy)^2+(z1-dz)^2)")
    #translation_restraint.addCollectiveVariable('translation', translation_restraint)
    translation_restraint.addGroup(atom_idxs)
    translation_restraint.addGlobalParameter("dx", dummy_atom_pos[0])
    translation_restraint.addGlobalParameter("dy", dummy_atom_pos[1])
    translation_restraint.addGlobalParameter("dz", dummy_atom_pos[2])
    translation_restraint.addGlobalParameter("kt", force_const)
    translation_restraint.addBond([0], [])
    #translation_restraint.setUsesPeriodicBoundaryConditions(True)
    return translation_restraint

def Orientaion_restraint(ref_pos, atom_idxs, qidx, center=0, force_const=8368*unit.kilojoule_per_mole/unit.nanometer**2):
    labels = ['x', 'y', 'z', 'w']
    q_cv = QuaternionForce(ref_pos, atom_idxs, qidx)
    k = f'kq{labels[qidx]}'
    q0 = f'q0{labels[qidx]}'
    q = f'q{labels[qidx]}'
    harmonic_energy_exp = f"0.5*{k}*({q}-{q0})^2"
    harmonic_restraint_force = omm.CustomCVForce(harmonic_energy_exp)
    harmonic_restraint_force.addCollectiveVariable(q, q_cv)
    harmonic_restraint_force.addGlobalParameter(q0, center)
    harmonic_restraint_force.addGlobalParameter(k, force_const)
    return harmonic_restraint_force

    # q0 = QuaternionForce(ref_pos, atom_idxs, 0)
    # q1 = QuaternionForce(ref_pos, atom_idxs, 1)
    # q2 = QuaternionForce(ref_pos, atom_idxs, 2)
    # q3 = QuaternionForce(ref_pos, atom_idxs, 2)
    # k = f'kq{labels[qidx]}'
    # q0 = f'q0{labels[qidx]}'
    # q = f'q{labels[qidx]}'
    # harmonic_energy_exp = f"0.5*kq0(q0)"

