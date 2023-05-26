import openmm.app as omma
import openmm as omm

import simtk.unit as unit
from Quaternionplugin import QuaternionForce
from Euleranglesplugin import EuleranglesForce

def RMSD_wall(ref_pos, atom_idxs, lowerwall=0.0*unit.nanometer, upperwall=0.3*unit.nanometer, force_const=2000*unit.kilojoule_per_mole/unit.nanometer**2):

    rmsd_cv = omm.RMSDForce(ref_pos, atom_idxs)
    # apply restraint if rmsd < lowerwall or rmsd > upperwall
    wall_energy_exp = "0.5*krmsd*(min(0,RMSD-lowerwall)^2+max(0,RMSD-upperwall)^2)"
    wall_restraint_force = omm.CustomCVForce(wall_energy_exp)
    wall_restraint_force.addCollectiveVariable("RMSD",  rmsd_cv)
    wall_restraint_force.addGlobalParameter("lowerwall", lowerwall)
    wall_restraint_force.addGlobalParameter("upperwall", upperwall)
    wall_restraint_force.addGlobalParameter("krmsd", force_const)
    
    return wall_restraint_force

def EulerAngle_wall(ref_pos, atom_idxs, fitatom_idxs, angle="Theta", lowerwall=-15.0, upperwall=15.0, force_const=100*unit.kilojoule_per_mole/unit.degree**2):
    eulerangle_cv = EuleranglesForce(ref_pos, atom_idxs, fitatom_idxs, angle)
    wall_energy_exp = "0.5*keulerangle*(min(0,angle-lowerwall)^2+max(0,angle-upperwall)^2)"
    wall_restraint_force = omm.CustomCVForce(wall_energy_exp)
    wall_restraint_force.addCollectiveVariable("angle",  eulerangle_cv)
    wall_restraint_force.addGlobalParameter("lowerwall", lowerwall)
    wall_restraint_force.addGlobalParameter("upperwall", upperwall)
    wall_restraint_force.addGlobalParameter("keulerangle", force_const)
    return wall_restraint_force

def EulerAngle_harmonic(ref_pos, atom_idxs, fitatom_idxs, center, angle="Theta", force_const=4184):
    eulerangle_cv = EuleranglesForce(ref_pos, atom_idxs, fitatom_idxs, angle)
    energy_exp = f"0.5*kheuler{angle}*(angle-anglecenter)^2"
    restraint_force = omm.CustomCVForce(energy_exp)
    restraint_force.addCollectiveVariable("angle",  eulerangle_cv)
    restraint_force.addGlobalParameter("anglecenter", center)
    restraint_force.addGlobalParameter(f"kheuler{angle}", force_const)
    return restraint_force

def RMSD_harmonic(ref_pos, atom_idxs, center, force_const=4184*unit.kilojoule_per_mole/unit.nanometer**2):
    rmsd_cv = omm.RMSDForce(ref_pos, atom_idxs)
    energy_exp = "0.5*khrmsd*(RMSD-RMSDcenter)^2"
    restraint_force = omm.CustomCVForce(energy_exp)
    restraint_force.addCollectiveVariable("RMSD",  rmsd_cv)
    restraint_force.addGlobalParameter("RMSDcenter", center)
    restraint_force.addGlobalParameter("khrmsd", force_const)

    return restraint_force
    
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

def q_restraint(ref_pos, atom_idxs, qidx, center=0, force_const=8368*unit.kilojoule_per_mole/unit.nanometer**2):
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

def Orientaion_restraint(ref_pos, atom_idxs, centers, force_consts):
    labels = ['x', 'y', 'z', 'w']
    
    harmonic_energy_exps = []
    for qidx in range(len(centers)):
        k = f'kq{labels[qidx]}'
        q0 = f'q0{labels[qidx]}'
        q = f'q{labels[qidx]}'
        harmonic_energy_exps.append(f"0.5*{k}*({q}-{q0})^2")
        # print(harmonic_energy_exps)
   
    harmonic_energy_exp = "+".join(harmonic_energy_exps)
    harmonic_restraint_force = omm.CustomCVForce(harmonic_energy_exp)
    for qidx in range(len(centers)):
        q_cv = QuaternionForce(ref_pos, atom_idxs, qidx)
        harmonic_restraint_force.addCollectiveVariable(f'q{labels[qidx]}', q_cv)
        harmonic_restraint_force.addGlobalParameter(f'q0{labels[qidx]}', centers[qidx])
        harmonic_restraint_force.addGlobalParameter(f'kq{labels[qidx]}', force_consts[qidx])
    return harmonic_restraint_force
