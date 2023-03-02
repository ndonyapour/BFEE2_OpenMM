/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2015 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed-code.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */


#include "qrotation.h"
#include "core/PlumedMain.h"
#include "tools/Vector.h"
#include <string>
#include <sstream> //For parsing the norm quaternion.
#include <cmath>

namespace PLMD{

//void qrotation(void)
qrotation::qrotation(void){
    q.zero();
    pos1_gradients = pos2_gradients = false;

}
void qrotation::build_correlation_matrix(const std::vector<Vector> pos1, const std::vector<Vector> pos2){

    C.zero();
    unsigned i;

    for(i=0; i<pos1.size(); i++){
        C += Tensor(pos1[i], pos2[i]);
    }
}
void qrotation::calculate_overlap_matrix(void){

    S = Matrix<double>(4, 4);

    S[0][0] =  - C[0][0] - C[1][1]- C[2][2];
    S[1][1] = - C[0][0] + C[1][1] + C[2][2];
    S[2][2] =  C[0][0] - C[1][1] + C[2][2];
    S[3][3] =  C[0][0] + C[1][1] - C[2][2];
    S[0][1] = - C[1][2] + C[2][1];
    S[0][2] = C[0][2] - C[2][0];
    S[0][3] = - C[0][1] + C[1][0];
    S[1][2] = - C[0][1] - C[1][0];
    S[1][3] = - C[0][2] - C[2][0];
    S[2][3] = - C[1][2] - C[2][1];
    S[1][0] = S[0][1];
    S[2][0] = S[0][2];
    S[2][1] = S[1][2];
    S[3][0] = S[0][3];
    S[3][1] = S[1][3];
    S[3][2] = S[2][3];

}
void  qrotation::diagonalize_matrix(const Vector4d normquat)
{
    int diagerror = diagMat(S, S_eigval, S_eigvec);
    if (diagerror!=0){
        std::string sdiagerror;
        Tools::convert(diagerror, sdiagerror);
        std::string msg="DIAGONALIZATION FAILED WITH ERROR CODE "+sdiagerror;
        plumed_merror(msg);
    }

    double dot;
    //Normalise each eigenvector in the direction closer to norm
    for (unsigned i=0;i<4;i++) {
        dot=0.0;
        for (unsigned j=0;j<4;j++) {
            dot += normquat[j] * S_eigvec[i][j];
        }
        if (dot < 0.0)
            for (unsigned j=0;j<4;j++)
                S_eigvec[i][j] =- S_eigvec[i][j];
    }

}

Vector4d qrotation::getQfromEigenvecs(unsigned idx){
    return Vector4d(S_eigvec[idx][0], S_eigvec[idx][1], S_eigvec[idx][2], S_eigvec[idx][3]);
}

void qrotation::request_group1_gradients(unsigned n){
    dS_1.resize(n, Matrix< Vector >(4, 4));
    dL0_1.resize(n, Vector(0.0, 0.0, 0.0));
    dQ0_1.resize(n, vector< Vector >(4));
    pos1_gradients = true;
}

void qrotation::request_group2_gradients(unsigned n){
    dS_2.resize(n, Matrix< Vector >(4, 4));
    dL0_2.resize(n, Vector(0.0, 0.0, 0.0));
    dQ0_2.resize(n, vector< Vector >(4));
    pos2_gradients = true;
}


// From NAMD
void qrotation::calc_optimal_rotation(const std::vector<Vector> pos1, const std::vector<Vector> pos2, const Vector4d normquat){

    q.zero();
    build_correlation_matrix(pos1, pos2);
    calculate_overlap_matrix();
    diagonalize_matrix(normquat);

    double const L0 = S_eigval[0];
    double const L1 = S_eigval[1];
    double const L2 = S_eigval[2];
    double const L3 = S_eigval[3];

    Vector4d const Q0 = getQfromEigenvecs(0);
    Vector4d const Q1 = getQfromEigenvecs(1);
    Vector4d const Q2 = getQfromEigenvecs(2);
    Vector4d const Q3 = getQfromEigenvecs(3);

    lambda = L0;
    q = Q0;

    q0 = q[0];  q1 = q[1]; q2 = q[2]; q3 = q[3];

    if (pos1_gradients){
    for (unsigned ia=0; ia < dS_1.size(); ia++) {
        //if (refw1[ia]==0) continue; //Only apply forces to weighted atoms in the RMSD calculation.

        double const rx = pos2[ia][0];
        double const ry = pos2[ia][1];
        double const rz = pos2[ia][2];

        Matrix < Vector >  &ds_1 = dS_1[ia];

        ds_1[0][0] = Vector(  rx,  ry,  rz);
        ds_1[1][0] = Vector( 0.0, -rz,  ry);
        ds_1[0][1] = ds_1[1][0];
        ds_1[2][0] = Vector(  rz, 0.0, -rx);
        ds_1[0][2] = ds_1[2][0];
        ds_1[3][0] = Vector( -ry,  rx, 0.0);
        ds_1[0][3] = ds_1[3][0];
        ds_1[1][1] = Vector(  rx, -ry, -rz);
        ds_1[2][1] = Vector(  ry,  rx, 0.0);
        ds_1[1][2] = ds_1[2][1];
        ds_1[3][1] = Vector(  rz, 0.0,  rx);
        ds_1[1][3] = ds_1[3][1];
        ds_1[2][2] = Vector( -rx,  ry, -rz);
        ds_1[3][2] = Vector( 0.0,  rz,  ry);
        ds_1[2][3] = ds_1[3][2];
        ds_1[3][3] = Vector( -rx, -ry,  rz);

        Vector                &dl0_1 = dL0_1[ia];
        vector<Vector>        &dq0_1 = dQ0_1[ia];

        for (unsigned i = 0; i < 4; i++) {
            for (unsigned j = 0; j < 4; j++) {
                dl0_1 += -1 * (Q0[i] * ds_1[i][j] * Q0[j]);
            }
        }
        for (unsigned p=0; p<4; p++) {
            for (unsigned i=0 ;i<4; i++) {
                for (unsigned j=0; j<4; j++) {
                    dq0_1[p] += -1 * (
                            (Q1[i] * ds_1[i][j] * Q0[j]) / (L0-L1) * Q1[p] +
                            (Q2[i] * ds_1[i][j] * Q0[j]) / (L0-L2) * Q2[p] +
                            (Q3[i] * ds_1[i][j] * Q0[j]) / (L0-L3) * Q3[p]);
                    }
                }
            }
        } // First loop

    }
    if (pos2_gradients) {
    for (unsigned ia=0; ia < dS_2.size(); ia++) {
        //if (refw1[ia]==0) continue; //Only apply forces to weighted atoms in the RMSD calculation.

        double const rx = pos1[ia][0];
        double const ry = pos1[ia][1];
        double const rz = pos1[ia][2];

        Matrix < Vector >  &ds_2 = dS_2[ia];

        ds_2[0][0] = Vector(  rx,  ry,  rz);
        ds_2[1][0] = Vector( 0.0, -rz,  ry);
        ds_2[0][1] = ds_2[1][0];
        ds_2[2][0] = Vector(  rz, 0.0, -rx);
        ds_2[0][2] = ds_2[2][0];
        ds_2[3][0] = Vector( -ry,  rx, 0.0);
        ds_2[0][3] = ds_2[3][0];
        ds_2[1][1] = Vector(  rx, -ry, -rz);
        ds_2[2][1] = Vector(  ry,  rx, 0.0);
        ds_2[1][2] = ds_2[2][1];
        ds_2[3][1] = Vector(  rz, 0.0,  rx);
        ds_2[1][3] = ds_2[3][1];
        ds_2[2][2] = Vector( -rx,  ry, -rz);
        ds_2[3][2] = Vector( 0.0,  rz,  ry);
        ds_2[2][3] = ds_2[3][2];
        ds_2[3][3] = Vector( -rx, -ry,  rz);

        Vector                &dl0_2 = dL0_2[ia];
        vector<Vector>        &dq0_2 = dQ0_2[ia];

        for (unsigned i = 0; i < 4; i++) {
            for (unsigned j = 0; j < 4; j++) {
                dl0_2 += -1 * (Q0[i] * ds_2[i][j] * Q0[j]);
            }
        }
        for (unsigned p=0; p<4; p++) {
            for (unsigned i=0 ;i<4; i++) {
                for (unsigned j=0; j<4; j++) {
                    dq0_2[p] += -1 * (
                            (Q1[i] * ds_2[i][j] * Q0[j]) / (L0-L1) * Q1[p] +
                            (Q2[i] * ds_2[i][j] * Q0[j]) / (L0-L2) * Q2[p] +
                            (Q3[i] * ds_2[i][j] * Q0[j]) / (L0-L3) * Q3[p]);
                    }
                }
            }
        } // Second loop
    }

    }

// From NAMD
Vector4d qrotation::position_derivative_inner(const Vector &pos, const Vector &vec)
{
  Vector4d result(0.0, 0.0, 0.0, 0.0);


  result[0] =   2.0 * pos[0] * q0 * vec[0]
               +2.0 * pos[1] * q0 * vec[1]
               +2.0 * pos[2] * q0 * vec[2]

               -2.0 * pos[1] * q3 * vec[0]
               +2.0 * pos[2] * q2 * vec[0]

               +2.0 * pos[0] * q3 * vec[1]
               -2.0 * pos[2] * q1 * vec[1]

               -2.0 * pos[0] * q2 * vec[2]
               +2.0 * pos[1] * q1 * vec[2];


  result[1] =  +2.0 * pos[0] * q1 * vec[0]
               -2.0 * pos[1] * q1 * vec[1]
               -2.0 * pos[2] * q1 * vec[2]

               +2.0 * pos[1] * q2 * vec[0]
               +2.0 * pos[2] * q3 * vec[0]

               +2.0 * pos[0] * q2 * vec[1]
               -2.0 * pos[2] * q0 * vec[1]

               +2.0 * pos[0] * q3 * vec[2]
               +2.0 * pos[1] * q0 * vec[2];


  result[2] =  -2.0 * pos[0] * q2 * vec[0]
               +2.0 * pos[1] * q2 * vec[1]
               -2.0 * pos[2] * q2 * vec[2]

               +2.0 * pos[1] * q1 * vec[0]
               +2.0 * pos[2] * q0 * vec[0]

               +2.0 * pos[0] * q1 * vec[1]
               +2.0 * pos[2] * q3 * vec[1]

               -2.0 * pos[0] * q0 * vec[2]
               +2.0 * pos[1] * q3 * vec[2];


  result[3] =  -2.0 * pos[0] * q3 * vec[0]
               -2.0 * pos[1] * q3 * vec[1]
               +2.0 * pos[2] * q3 * vec[2]

               -2.0 * pos[1] * q0 * vec[0]
               +2.0 * pos[2] * q1 * vec[0]

               +2.0 * pos[0] * q0 * vec[1]
               +2.0 * pos[2] * q2 * vec[1]

               +2.0 * pos[0] * q1 * vec[2]
               +2.0 * pos[1] * q2 * vec[2];

  return result;
}

// Vector4d qrotation::conjugat(void){

//     return Vector4d(q0, -q1, -q2, -q3);
// }
// Vector qrotation::rotate(const Vector vec){

//     Vector4d qc, vec4d, result;
//     vec4d = Vector4d(0.0, vec[0], vec[1], vec[2]);
//     result = Vector4d(0.0, 0.0, 0.0, 0.0);
//     qc = this->conjugat();
//     for (unsigned i=0; i<4; i++){
//         result[i] = q[i] * vec[i] * qc[i];
//     }
//     return Vector(result[1], result[2], result[3]);
// }

std::vector<Vector> qrotation::rotateCoordinates(Vector4d qr, const std::vector<Vector> pos){
    std::vector<Vector> rot_pos;
    unsigned ntot = pos.size();
    //rot_pos.resize(ntot, Vector(0,0,0));
    for (unsigned i=0; i<ntot; i++)
            rot_pos.push_back((this->quaternionRotate(qr, pos[i])));

    return rot_pos;
}

Vector4d qrotation::quaternionInvert(const Vector4d q){
    return Vector4d(q[0],-q[1],-q[2],-q[3]);
}


Vector4d qrotation::quaternionProduct(const Vector4d& v1,const Vector4d& v2){
    return Vector4d(
        v1[0]*v2[0]-v1[1]*v2[1]-v1[2]*v2[2]-v1[3]*v2[3],
        v1[0]*v2[1]+v1[1]*v2[0]+v1[2]*v2[3]-v1[3]*v2[2],
        v1[0]*v2[2]+v1[2]*v2[0]+v1[3]*v2[1]-v1[1]*v2[3],
        v1[0]*v2[3]+v1[3]*v2[0]+v1[1]*v2[2]-v1[2]*v2[1]);
}


Vector qrotation::quaternionRotate(const Vector4d& qq, const Vector& v){
    double q0 = qq[0];
    Vector vq = Vector(qq[1],qq[2],qq[3]);
    Vector a;
    Vector b;
    a = crossProduct(vq, v)+ q0 * v;
    b = crossProduct(vq, a);
    return b+b+v;
}

}

