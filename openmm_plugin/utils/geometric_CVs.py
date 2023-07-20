import os
import numpy as np
import math

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
            print("I'm here")
            # second group used for fitting 
            # center ref pos
            # import ipdb 
            # ipdb.set_trace()
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
        qs = []
        enable_fitting = False if fittingGroup_idxs is None else True
        radian_to_degree = 180 / 3.1415926
        for idx in range(positions.shape[0]):
            rot_q = self.calc_quaternion(refpositions[group_idxs], positions[idx][group_idxs], 
                                         refpositions[fittingGroup_idxs], positions[idx][fittingGroup_idxs], 
                                         enable_fitting)
            # if idx == 114:
            #     import ipdb
            #     ipdb.set_trace()
            #if (angle == "Theta") {
            qs.append(rot_q)
            q1 = rot_q[0]  
            q2 = rot_q[1] 
            q3 = rot_q[2] 
            q4 = rot_q[3]
            x = 2 * (q1 * q3 - q4 * q2)
            angles.append(radian_to_degree * math.asin(x))
        return np.array(angles)



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
        from scipy.linalg import eig
        self.S_eigval, self.S_eigvec = np.linalg.eig(S)
        print(self.S_eigvec)
        # print("************************")

        # convert complex values to real
        # self.S_eigval = np.real(self.S_eigval)
        # self.S_eigvec = np.real(self.S_eigvec)

    def getQfromEigenvecs(self, idx):
        # Note the first column is not the first eigen vector
        eig_idx = np.argmax(np.abs(self.S_eigvec[0, :]))
        self.Swap(self.S_eigvec, 0, eig_idx)
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

