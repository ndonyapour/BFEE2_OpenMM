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

#ifndef __PLUMED_tools_qrotation_h
#define __PLUMED_tools_qrotation_h

#include "core/PlumedMain.h"
#include "tools/PDB.h"
#include "core/Atoms.h"

//#define DEBUG__CHENP
// Todo make derivates optional, turned off for now

using namespace std;
namespace PLMD{

class Log;
class PDB;

class qrotation {
    public:

    Matrix< double > S, S_eigvec;
    Tensor C;
    std::vector< double > S_eigval;
    Vector4d q;
    double lambda;
    double q0, q1, q2, q3;

    /// Derivatives of S
    std::vector<Matrix< Vector > > dS_1,  dS_2;
    /// Derivatives of leading eigenvalue
    std::vector< Vector >  dL0_1, dL0_2;
    /// Derivatives of leading eigenvector
    std::vector< std::vector<Vector > > dQ0_1, dQ0_2;
    explicit qrotation(void);
    void calc_optimal_rotation(const std::vector<Vector> pos1, const std::vector<Vector> pos2, const Vector4d normquat);
    void build_correlation_matrix(const std::vector<Vector> pos1, const std::vector<Vector> pos2);
    void calculate_overlap_matrix(void);
    void diagonalize_matrix(const Vector4d normquat);
    Vector4d getQfromEigenvecs(unsigned idx);
    void request_group1_gradients(unsigned n);
    void request_group2_gradients(unsigned n);
    Vector4d position_derivative_inner(const Vector &pos, const Vector &vec);
    Vector4d conjugat(void);
    Vector rotate(const Vector vec);
    std::vector<Vector> rotateCoordinates(const Vector4d qr, const std::vector<Vector> pos);
    Vector4d quaternionInvert(const Vector4d q);
    Vector4d quaternionProduct(const Vector4d& v1, const Vector4d& v2);
    Vector quaternionRotate(const Vector4d& qq, const Vector& v);
    private:
    bool pos1_gradients, pos2_gradients;

    //void calculate_gradients(const std::vector<Vector> pos);

};

}

#endif

