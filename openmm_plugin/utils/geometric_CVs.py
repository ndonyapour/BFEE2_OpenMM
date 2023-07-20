import os
import numpy as np
import math

import mdtraj as mdj

def calculateCOG(pos):
    return np.mean(pos, axis=0)

def translateCoordinates(pos, t):
    return pos - t

def shiftbyCOG(pos):
    cog = calculateCOG(pos)
    return translateCoordinates(pos, cog)

class GeometricCVs:

    def __init__(self):
        pass

    def calc_quaternion(self, refpositions, positions, fitting_refpositions=[], fitting_positions=[], fitting=False):

        rot_q = None
        if not fitting:
            # Cenetr groupA pos
            centered_pos = shiftbyCOG(positions)

            # Center refpos group A
            centerded_refpos = shiftbyCOG(refpositions)
            quaternion = Quaternion()                            
            rot_q = quaternion.calc_optimal_rotation(centerded_refpos, centered_pos)
            return rot_q

        else:
            centered_refpos = shiftbyCOG(refpositions)
            centered_refpos_fitgroup = shiftbyCOG(fitting_refpositions)

            # calc COG of the fitting group
            fitgroup_COG = calculateCOG(fitting_positions)
            centered_pos = translateCoordinates(positions, fitgroup_COG)
            centered_pos_fitgroup = translateCoordinates(fitting_positions, fitgroup_COG)
            quaternion = Quaternion()
            fit_rot_q = quaternion.calc_optimal_rotation(centered_pos_fitgroup, centered_refpos_fitgroup)

            # # rotate both groups
            quaternion = Quaternion()
            rotated_pos = quaternion.rotateCoordinates(fit_rot_q, centered_pos)

            # find optimal rotation between aligned group atoms 
            rot_q = quaternion.calc_optimal_rotation(centered_refpos, rotated_pos)
            #print(rot_q)

            return rot_q

    def EuelrAngle(self, refpositions, positions, group_idxs, fittingGroup_idxs=None, angle='Theta'):
        angles = []
        #qs = []
        enable_fitting = False if fittingGroup_idxs is None else True
        radian_to_degree = 180 / 3.1415926
        for idx in range(positions.shape[0]):
            rot_q = self.calc_quaternion(refpositions[group_idxs], positions[idx][group_idxs], 
                                         refpositions[fittingGroup_idxs], positions[idx][fittingGroup_idxs], 
                                         enable_fitting)
   
            q1 = rot_q[0]  
            q2 = rot_q[1] 
            q3 = rot_q[2] 
            q4 = rot_q[3]
            
            if angle=='Theta':
                x = 2 * (q1 * q3 - q4 * q2)
                angles.append(radian_to_degree * math.asin(x))
            elif angle=='Phi':
                x = 2*(q1*q2+q3*q4)
                y = 1-2*(q2*q2+q3*q3)
                angles.append(radian_to_degree * math.atan2(x, y))
            elif angle=='Psi':
                x = 2*(q1*q4+q2*q3)
                y = 1-2*(q3*q3+q4*q4)
                angles.append(radian_to_degree * math.atan2(x, y))
            elif angle=='all':
                x = 2 * (q1 * q3 - q4 * q2)
                theta = radian_to_degree * math.asin(x)
                x = 2*(q1*q2+q3*q4)
                y = 1-2*(q2*q2+q3*q3)
                phi= radian_to_degree * math.atan2(x, y)
                x = 2*(q1*q4+q2*q3)
                y = 1-2*(q3*q3+q4*q4)
                psi = radian_to_degree * math.atan2(x, y)
                angles.append([theta, phi, psi])
            else:
                print("The angle type is wrong")
                exit()
        return np.array(angles)
    
    def PolarAngle(self, refpositions, positions, group_idxs, fittingGroup_idxs, angle='Theta'):
        angles = []
        #qs = []
        enable_fitting = False if fittingGroup_idxs is None else True
        radian_to_degree = 180 / 3.1415926
        for idx in range(positions.shape[0]):
            # centered_refpos = shiftbyCOG(refpositions[group_idxs])
            centered_refpos_fitgroup = shiftbyCOG(refpositions[fittingGroup_idxs])

            # calc COG of the fitting group
            fitgroup_COG = calculateCOG(positions[idx][fittingGroup_idxs])
            centered_pos = translateCoordinates(positions[idx][group_idxs], fitgroup_COG)
            centered_pos_fitgroup = translateCoordinates(positions[idx][fittingGroup_idxs], fitgroup_COG)
            quaternion = Quaternion()
            fit_rot_q = quaternion.calc_optimal_rotation(centered_pos_fitgroup, centered_refpos_fitgroup)
            rotated_pos = quaternion.rotateCoordinates(fit_rot_q, centered_pos)
            rotated_fitpos = quaternion.rotateCoordinates(fit_rot_q, centered_pos_fitgroup)
   
                
            fit_rot_pos_cog = calculateCOG(rotated_fitpos)
            rot_pos_cog = calculateCOG(rotated_pos)
            distance = rot_pos_cog - fit_rot_pos_cog
            norm = np.sqrt(np.dot(distance, distance))
            unit_vec = distance / norm
            i1 = unit_vec[0]
            i2 = unit_vec[1]
            i3 = unit_vec[2]
            if angle == "Theta":
                angles.append(radian_to_degree * math.acos(-i2))

            elif angle=='Phi':
                angles.append(radian_to_degree * math.atan2(i3, i1))
   
            elif angle=='all':
                theta = radian_to_degree * math.acos(-i2)
                phi = radian_to_degree * math.atan2(i3, i1)
                angles.append([theta, phi])
            else:
                print("The angle type is wrong")
                exit()
        return np.array(angles)
    
    def rmsd(self, pdb, traj, atom_idxs):

        return mdj.rmsd(pdb, traj, atom_indices=atom_idxs)
    
    def r(self, traj, selection1, seletion2):
        coms1 = mdj.compute_center_of_mass(traj, select=selection1)
        coms2 = mdj.compute_center_of_mass(traj, select=seletion2)
        return np.linalg.norm(coms1 - coms2, axis=1)
    
    def translation(sef, ref_pdb, traj, selection):
        ref_com = mdj.compute_center_of_mass(ref_pdb, select=selection)
        coms = mdj.compute_center_of_mass(traj, select=selection)
        return np.linalg.norm(coms - ref_com, axis=1)

# np.max(translation_cv)


class Quaternion:
    """Finds the optimal rotation between two molecules using quaternion.
    The details of this method can be found on
    https://onlinelibrary.wiley.com/doi/10.1002/jcc.20110.
    """

    def __init__(self, normquat=(1.0, 0.0, 0.0, 0.0)):
        """Constructor for the Quaternion model
        """
        self.is_trainable = False
        self.normquat = normquat
        self.S = np.zeros((4, 4))
        self.S_eigval = np.zeros(4)
        self.S_eigvec = np.zeros((4, 4))
        self.q = np.zeros(4)
      

    def build_correlation_matrix(self, pos1, pos2):
        # C = np.zeros((3, 3))
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(pos1.shape[0]):
        #             C[i][j] += pos1[k][i]*pos2[k][j]

        return np.matmul(pos1.T, pos2)

    def calculate_overlap_matrix(self, C):

        S = np.zeros((4, 4))

        # S[0][0] = C[0][0] + C[1][1] + C[2][2]
        # S[1][1] = C[0][0] - C[1][1] - C[2][2]
        # S[2][2] = - C[0][0] + C[1][1] - C[2][2]
        # S[3][3] = - C[0][0] - C[1][1] + C[2][2]
        # S[0][1] = C[1][2] - C[2][1]
        # S[0][2] = - C[0][2] + C[2][0]
        # S[0][3] = C[0][1] - C[1][0]
        # S[1][2] = C[0][1] + C[1][0]
        # S[1][3] = C[0][2] + C[2][0]
        # S[2][3] = C[1][2] + C[2][1]
        # S[1][0] = S[0][1]
        # S[2][0] = S[0][2]
        # S[2][1] = S[1][2]
        # S[3][0] = S[0][3]
        # S[3][1] = S[1][3]
        # S[3][2] = S[2][3]
        S[0][0] =  - C[0][0] - C[1][1]- C[2][2]
        S[1][1] = - C[0][0] + C[1][1] + C[2][2]
        S[2][2] =  C[0][0] - C[1][1] + C[2][2]
        S[3][3] =  C[0][0] + C[1][1] - C[2][2]
        S[0][1] = - C[1][2] + C[2][1]
        S[0][2] = C[0][2] - C[2][0]
        S[0][3] = - C[0][1] + C[1][0]
        S[1][2] = - C[0][1] - C[1][0]
        S[1][3] = - C[0][2] - C[2][0]
        S[2][3] = - C[1][2] - C[2][1]
        S[1][0] = S[0][1]
        S[2][0] = S[0][2]
        S[2][1] = S[1][2]
        S[3][0] = S[0][3]
        S[3][1] = S[1][3]
        S[3][2] = S[2][3]

        self.S = S
        return S
    def Swap(self,  arr, start_index, last_index):
            arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]]

    def diagonalize_matrix(self, S):
        # U, S, V = np.linalg.svd(self.S)
        # self.S_eigval = S**2 / (S**2).sum()
        # self.S_eigvec = V
        self.S_eigval, self.S_eigvec = np.linalg.eig(S)
        # print("************************")

        # convert complex values to real
        self.S_eigval = np.real(self.S_eigval)
        self.S_eigvec = np.real(self.S_eigvec)

    def getQfromEigenvecs(self, idx):
        # Note the first column is not the first eigen vector
        eig_idx = np.argmax(np.abs(self.S_eigvec[0, :]))
        self.Swap(self.S_eigvec, 0, eig_idx)
        self.S_eigval[0], self.S_eigval[eig_idx] =  self.S_eigval[eig_idx], self.S_eigval[0]
        normquat = np.array(self.normquat)
        if np.matmul(self.S_eigvec[:, idx], normquat) < 0:
            return -1 * self.S_eigvec[:, idx]
        else:
            return  self.S_eigvec[:, idx]

    def calc_optimal_rotation(self, pos1, pos2):
        C = self.build_correlation_matrix(pos1, pos2)
        S = self.calculate_overlap_matrix(C)
        self.diagonalize_matrix(S)
        self.q = self.getQfromEigenvecs(0)
        return self.q

    def rotateCoordinates(self, q, pos):
        return np.stack([self.quaternionRotate(q, row) for row in pos])

    def quaternionInvert(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)

    def quaternionRotate(self, q, vec):
        q0 = q[0]
        vq = np.array([q[1], q[2], q[3]])
        a = np.cross(vq, vec) + q0 * vec
        b = np.cross(vq, a)
        # a = np.cross(vec, vq) + q0 * vec
        # b = np.cross(a, vq)
        return b + b + vec
    def getEigenValue(self, idx):
        return self.S_eigval[idx]

