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
#include "colvar/Colvar.h"
#include "qrotation.h"
#include "core/PlumedMain.h"
#include "colvar/ActionRegister.h"
#include "tools/PDB.h"
#include "core/Atoms.h"

#include <string>
#include <sstream> //For parsing the norm quaternion.
#include <cmath>

//#define DEBUG__CHENP
// Todo make derivates optional, turned off for now

using namespace std;

namespace PLMD{
namespace colvar{

//+PLUMEDOC COLVAR DistanceDir
/*
 * Calculates DistanceDir rotation to a reference.
 *
 * Version: 0.2 - Poker Chen 07.01.2016
 * Added relative DistanceDir between two domains.
 *
 * Version: 0.1 - Poker Chen 12.11.2015
 * This functionality is intended as a near-clone to RMSD,
 * in which one can restrain the orientation of a molecule in an arbitrary
 * rotation relative to a reference frame, or relative orientation of
 * two molecules, given respective reference coordinates that
 * would place them in the same frame.
 *
 * There are several KEYWORDS that are necessary. One must have wither one or two
 * reference coordinate files, with atoms to be fitted having an occupancy of > 0.0.
 * If only one is given in REFERENCE, then the absolute rotation with the system frame
 * is calculated.
 * If both REFERENCE and GROUPB_REFERENCE are given, then the relative orientation
 * between the two groups are given is calculated, using the convention
 *  qB = qr.qA, or, qr=qB.qA^-1 = qB.(-qA)
 *
 * NORM_DIRECTION can also be set to remove the 2-fold degeneracy of q==-q,
 * When using restraints, this should usually be set to the restraint DistanceDir itself.
 * PLUMED will automatically normalise this, so you can use (1,1,1,1) -> (0.5,0.5,0.5,0.5)
 *
\par Examples

i: DISTANCEDIR GROUPA_REFERENCE=./refA.pdb GROUPB_REFERENCE=./refB.pdb

RESTRAINT ARG=q.w,q[0],q[1],q[2] AT=0.70710678118,0.70710678118,0,0 KAPPA=24.94,24.94,24.94,24.94 LABEL=restx

*/
//+ENDPLUMEDOC

//Declare class and content variables
class DistanceDir : public Colvar {
    private:
    unsigned int refnat1;
    std::vector<AtomNumber> refind1;
    std::vector<Vector> refposA;
    std::vector<double> refw1;
    std::vector<double> refdisp1;
    Vector refc1;
    bool bRefc1_is_calculated;
    bool bRef1_is_centered;
    bool NoFitGradients;

    //PLMD::RMSDBase* rmsd2;
    unsigned int refnat2;
    std::vector<AtomNumber> refind2;
    std::vector<Vector> refposB;
    std::vector<double> refw2;
    std::vector<double> refdisp2;
    Vector refc2;
    bool bRefc2_is_calculated;
    bool bRef2_is_centered;
    Matrix<double> S12;
    Vector4d quat2;
    double lambda12;
    bool pbc;

    // Relevant variables for outputs and derivatives.
    double dist;
    double lambda;

    // Local Variables for rotational optimisation.
    Matrix<double> Smat;
    std::vector<double> eigenvals;
    Matrix<double> qmat;

    Vector4d quat;
    Vector4d normquat; //Direction of normalisation.
    //std::vector<double> eigenvals;
    //Matrix<double> eigenvecs;
    double rr00; //  sum of positions squared (needed for dist calc)
    double rr11; //  sum of reference squared (needed for dist calc)
    Tensor rr01;
    Tensor rotation; // rotation derived from the eigenvector having the smallest eigenvalue
    Tensor drotation_drr01[3][3]; // derivative of the rotation only available when align!=displace
    Tensor ddist_drr01;
    Tensor ddist_drotation;
    std::vector<Vector> diff; // difference of components

public:
  explicit DistanceDir(const ActionOptions&);
  ~DistanceDir();
  void clear();
  // set reference coordinates, remove the com by using uniform weights
  void setReferenceA(const std::vector<AtomNumber> & inpind, const std::vector<Vector> & inppos,
                     const std::vector<double> & inpw, const std::vector<double> & inpdisp );
  void setReferenceB(const std::vector<AtomNumber> & inpind, const std::vector<Vector> & inppos,
                     const std::vector<double> & inpw, const std::vector<double> & inpdisp );
  //void setReference2(const std::vector<Vector> & inppos, const std::vector<double> & inpw, const std::vector<double> & inpdisp );
  void setReferenceA(const PDB inppdb );
  void setReferenceB(const PDB inppdb );
  std::vector<Value*> setupDistanceDirColvarPtr(Vector q);

// active methods:
  //Because I don't have the time to fuck around with parseVector from Bias.
  Vector4d stringToQuat(const string str, const char *fs);

  void optimalAlignment1(const std::vector<Vector> & currpos);

  Vector calculateCOG(const std::vector<Vector> pos);
  std::vector<Vector> translateCoordinates(const std::vector<Vector> pos, const Vector t);

  virtual void calculate();
  static void registerKeywords(Keywords& keys);
};

//Functions to parse options and keywords in plumed.dat
PLUMED_REGISTER_ACTION(DistanceDir, "DISTANCEDIR")

void DistanceDir::registerKeywords(Keywords& keys){
  Colvar::registerKeywords(keys);
  keys.add("compulsory","GROUPA_REFERENCE","A file in pdb format containing the reference structure and the atoms involved in the CV.");
  keys.add("compulsory","GROUPB_REFERENCE","A second reference file for the second group to calculate relative orientation.");

  keys.addFlag("NoFitGradients", false, "Ignores the fiiting rotational gradients contribution to colvar gradients");
  //keys.remove("NOPBC");
  keys.addFlag("COMPONENTS", true, "(Compulsary) Calculate the DistanceDir as a vector, stored as label.x, label.y, label.z");
  keys.addOutputComponent("x", "COMPONENTS","the x-component of the normailized distance");
  keys.addOutputComponent("y", "COMPONENTS","the y-component of the normailized distance");
  keys.addOutputComponent("z", "COMPONENTS","the z-component of the normailized distance");
}

Vector4d DistanceDir::stringToQuat(const string str, const char* fs)
{
    //Use find to location positions of field separators
    //Cat each substring and give back q.
    //fprintf (stderr, "= = Debug: STRINGTOQUAT has been called..\n");
    std::stringstream ss(str);
    double val;
    Vector4d q;

    unsigned i=0;
    while (ss >> val)
    {
        if (i>3) break;
        q[i]=val;
        if (ss.peek() == *fs)
            ss.ignore();
        i++;
    }
    if (i!=4) fprintf(stderr,"= = = WARNING: did not read 4 components to the normalisation DistanceDir!\n");
    //fprintf (stderr, "= = Debug: finished STRINGTOQUAT..");
    //fprintf(stderr,"q: %g %g %g %g\n", q[0], q[1], q[2], q[3]);
    return q;
}

//Constructor
DistanceDir::DistanceDir(const ActionOptions&ao):
//PLUMED_COLVAR_INIT(ao)
//Initialise posre1 to hold 1 position and 0 derivatives.
//Initialise mypack to be empty with no arguments and no atoms.
//PLUMED_COLVAR_INIT(ao),posder1(1,0), mypack1(0,0,posder1), posder2(1,0)
PLUMED_COLVAR_INIT(ao),
pbc(true)
{
    fprintf(stderr, "= = Debug: Constructing the DistanceDir colvar...\n");


    string reffileA ,reffileB;

    parse("GROUPA_REFERENCE", reffileA);
    parse("GROUPB_REFERENCE", reffileB);
    string type;
    type.assign("OPTIMAL");
     parseFlag("NoFitGradients", NoFitGradients);
    bool nopbc=!pbc;
    parseFlag("NOPBC",nopbc);
    pbc=!nopbc;

    //Setup the normalisation direction of q. Initialise a vector
    //and then give its address to the COLVAR.
    Vector4d normquat(1.0, 0.0 ,0.0 , 0.0);
    checkRead();

    fprintf (stderr, "= = = Debug: Will now add components...\n");
    addComponentWithDerivatives("x"); componentIsNotPeriodic("x");
    addComponentWithDerivatives("y"); componentIsNotPeriodic("y");
    addComponentWithDerivatives("z"); componentIsNotPeriodic("z");
    //log<<"NOTE: q is stored as four components (w x y z).\n";

    PDB pdbA; //PDB storage of reference coordinates 1
    PDB pdbB; //PDB storage of reference coordinates 2

    fprintf (stderr, "= = = Debug: Will now read REFERENCE pdb file...\n");
    // read everything in ang and transform to nm if we are not in natural units
    if( !pdbA.read(reffileA,plumed.getAtoms().usingNaturalUnits(),0.1/atoms.getUnits().getLength()) )
        error("missing input file " + reffileA );

    //Store reference positions with zeroed barycenters.
    setReferenceA(pdbA);
    fprintf(stderr, "= = = = Debug: read and parse finished.\n");

    fprintf (stderr, "= = = Debug: GROUPB_REFERENCE found, parsing second pdb file...\n");
    if( !pdbB.read(reffileB,plumed.getAtoms().usingNaturalUnits(),0.1/atoms.getUnits().getLength()) )
    error("missing input file " + reffileB );

        //Store reference positions with zeroed barycenters.
    setReferenceB(pdbB);
    fprintf (stderr, "= = = = Debug: read and parse finished ...\n");

    //if (bRefc2_is_calculated) {
        // 2 Reference files, relative quaternions.
    std::vector<AtomNumber> atoms;
    fprintf(stderr,"= = = = Debug: First 3 entries of %lu pdbA atoms: %u %u %u\n",
            refind1.size(), refind1[0].index(), refind1[1].index(), refind1[2].index());
    fprintf(stderr,"= = = = Debug: First 3 entries of %lu pdbB atoms: %u %u %u\n",
            refind2.size(), refind2[0].index(), refind2[1].index(), refind2[2].index());

    atoms.reserve(refind1.size()+refind2.size());
    atoms.insert(atoms.end(), refind1.begin(), refind1.end());
    atoms.insert(atoms.end(), refind2.begin(), refind2.end());

    requestAtoms(atoms);
    log.printf("  reference from files %s and %s\n",
            reffileA.c_str(), reffileB.c_str());
    log.printf(" which contains %d atoms\n",getNumberOfAtoms());

    fprintf (stderr, "= = = = Debug: Request finished.\n");
    fprintf (stderr, "= = = Debug: finished constructing DistanceDir.\n");
    //atoms.clearFullList();
}

DistanceDir::~DistanceDir(){
  //delete rmsd1;
  //if (rmsd2) delete rmsd2;
}

void DistanceDir::clear(){
    refind1.clear();
    refposA.clear();
    refind2.clear();
    refposB.clear();
}

//Currently uses a C-like approach, should encapsulate into an object.
//Prefilters the atom list to remove all atoms with zero weight, reducing the passing of
//data between plumed and the MD engine.

void DistanceDir::setReferenceA(const std::vector<AtomNumber> & inpind, const std::vector<Vector> & inppos,
                               const std::vector<double> & inpw, const std::vector<double> & inpdisp ){
    unsigned nvals=inpind.size();
    plumed_massert(refw1.empty(),"you should first clear() an RMSD object, then set a new reference");
    plumed_massert(refdisp1.empty(),"you should first clear() an RMSD object, then set a new reference");


    refnat1=0;
    vector<unsigned> atom_internal_idxs;

    for (unsigned i=0;i<nvals;i++)
    {
        if (inpw[i]>0.0)
        {
            refnat1++;
            //fprintf(stderr,"DEBUG: %u %g %u\n", inpind[i].index(), inpw[i], refnat1);
            atom_internal_idxs.push_back(i);
            refind1.push_back(inpind[i]);
        }
    }

    fprintf(stderr," = = = DEBUG: found %u of %u atoms with non-zero weights.\n", refnat1, nvals);
    //Now efficiently add submatrix.
    refposA.reserve(refnat1);
    refw1.reserve(refnat1);
    refdisp1.reserve(refnat1);
    for (unsigned i=0; i<refnat1; i++)
    {
        //fprintf(stderr,"= = = DEBUG check for atom indices: %i ... %i\n", refind1[i].index(), i);
        refposA.push_back(inppos[atom_internal_idxs[i]]);
        refw1.push_back(inpw[atom_internal_idxs[i]]);
        refdisp1.push_back(inpdisp[atom_internal_idxs[i]]);
        //refind1[i]=inpind[i];
        //fprintf(stderr,"atom idxs... %u\n", atom_internal_idxs[i]);
    }

    // Only need to iterate over subindices.
    // we might need to change this
    double wtot=0.0, disptot=0.0;  //refw1[i]*
    for(unsigned i=0;i<refnat1;i++) {refc1+=refposA[i]; wtot+=refw1[i]; disptot+=refdisp1[i];}
    fprintf(stderr, "= = = = = Debug setReferenceA(): wtot is %g and disptot is %g\n", wtot, disptot);
    refc1/=refposA.size();
    for(unsigned i=0;i<refnat1;i++) {refposA[i]-=refc1; refw1[i]=refw1[i]/wtot ; refdisp1[i]=refdisp1[i]/disptot; }
    bRefc1_is_calculated=true;
    bRef1_is_centered=true;
}
void DistanceDir::setReferenceA(const PDB inppdb ){
    setReferenceA( inppdb.getAtomNumbers(), inppdb.getPositions(), inppdb.getOccupancy(), inppdb.getBeta() );
}

void DistanceDir::setReferenceB(const std::vector<AtomNumber> & inpind, const std::vector<Vector> & inppos,
                               const std::vector<double> & inpw, const std::vector<double> & inpdisp ){
    unsigned nvals=inpw.size();
    plumed_massert(refw2.empty(),"you should first clear() an RMSD object, then set a new reference");
    plumed_massert(refdisp2.empty(),"you should first clear() an RMSD object, then set a new reference");

    //Preallocate to maximum possible
    //refind2.reserve(nvals);

    //Filter out the zero weight data.
    refnat2=0;
    vector<unsigned> atom_internal_idxs;

    for (unsigned i=0;i<nvals;i++)
    {
        if (inpw[i]>0.0)
        {
            refnat2++;
            //fprintf(stderr,"DEBUG: %u %g %u\n", inpind[i].index(), inpw[i], refnat2);
            refind2.push_back(inpind[i]);
            atom_internal_idxs.push_back(i);
        }
    }
    fprintf(stderr," = = = DEBUG: found %u of %u atoms with non-zero weights.\n", refnat2, nvals);
    //Now efficiently populate submatrix.
    refposB.reserve(refnat2);
    refw2.reserve(refnat2);
    refdisp2.reserve(refnat2);
    for (unsigned i=0;i<refnat2;i++)
    {
        //fprintf(stderr,"= = = DEBUG check for atom indices: %i ... %i\n", refind2[i].index(), i);
        refposB.push_back(inppos[atom_internal_idxs[i]]);
        refw2.push_back(inpw[atom_internal_idxs[i]]);
        refdisp2.push_back(inpdisp[atom_internal_idxs[i]]);
        //refind2[i]=inpind[i];
        //fprintf(stderr,"... %i\n", refind2[i].index());
    }

    // Only need to iterate over subindices.
    double wtot=0.0, disptot=0.0;
    for(unsigned i=0;i<refnat2;i++) {refc2+=refposB[i]; wtot+=refw2[i]; disptot+=refdisp2[i];}
    fprintf(stderr, "= = = = = Debug setReferenceB(): wtot is %g and disptot is %g\n", wtot, disptot);
    refc2/=refposB.size();
    for(unsigned i=0;i<refnat2;i++) {refposB[i]-=refc2; refw2[i]=refw2[i]/wtot ; refdisp2[i]=refdisp2[i]/disptot; }
    bRefc2_is_calculated=true;
    bRef2_is_centered=true;
}

void DistanceDir::setReferenceB(const PDB inppdb ){
    setReferenceB( inppdb.getAtomNumbers(), inppdb.getPositions(), inppdb.getOccupancy(), inppdb.getBeta() );
}

Vector DistanceDir::calculateCOG(const std::vector<Vector> pos)
{
    Vector cog = Vector(0,0,0);
    for (unsigned i=0; i<pos.size(); i++)
            cog += pos[i];
    cog /= pos.size();
    return cog;
}

std::vector<Vector> DistanceDir::translateCoordinates(const std::vector<Vector> pos, const Vector t)
{
    std::vector<Vector> translated_pos;
    for (unsigned i=0; i<pos.size(); i++) {
        translated_pos.push_back(pos[i] - t);
    }
 return translated_pos;
}


std::vector<Value*> DistanceDir::setupDistanceDirColvarPtr(Vector q)
{

    Value* valuex=getPntrToComponent("x");
    Value* valuey=getPntrToComponent("y");
    Value* valuez=getPntrToComponent("z");
    std::vector<Value*> qptr;

    qptr.push_back(valuex);
    qptr.push_back(valuey);
    qptr.push_back(valuez);

    for (unsigned i=0;i<3;i++) qptr[i]->set(q[i]);

    return qptr;
}


void DistanceDir::optimalAlignment1(const std::vector<Vector> & currpos)
{


    std::vector<Vector> currposA, currposB, translate_posA, translate_posB, rot_posA,rot_posB;

    std::vector<Value*> distptr;
    qrotation rot;
    Vector distance, posA_cog, posB_cog;
    double norm;

    //currposA.resize(refposA.size(), Vector);
    for (unsigned i=0; i<refposA.size(); i++ )
        currposA.push_back(currpos[i]);

    for (unsigned i=0; i<refposB.size(); i++ )
        currposB.push_back(currpos[i+refposA.size()]);



    posA_cog = calculateCOG(currposA);

    translate_posA = translateCoordinates(currposA, posA_cog);
    translate_posB = translateCoordinates(currposB, posA_cog);

    // You need to request gradients first
    rot.request_group1_gradients(currposA.size());
    rot.calc_optimal_rotation(translate_posA, refposA, normquat);


    rot_posA = rot.rotateCoordinates(rot.q, translate_posA);
    rot_posB = rot.rotateCoordinates(rot.q, translate_posB);

    // calculate COG of each rotated pos
    Vector rot_fit_com, rot_pos_com;
    posA_cog = calculateCOG(rot_posA);
    posB_cog = calculateCOG(rot_posB);

    if(pbc){

        distance = pbcDistance(posA_cog, posB_cog); //posB_cog - posA_cog
    }
    else {

        distance = delta(posA_cog, posB_cog);
    }

    norm = distance.modulo();
    Vector unit_vec = distance / norm;


    distptr = setupDistanceDirColvarPtr(unit_vec);

    // derivative for posB
    //  setAtomsDerivatives (valuea,0,matmul(nor,Vector(-1,0,0)));
    // setAtomsDerivatives (valuea,1,matmul(getPbc().getInvBox(),Vector(+1,0,0)));
    // valuea->set(Tools::pbc(d[0]));
    // setAtomsDerivatives (valueb,0,matmul(getPbc().getInvBox(),Vector(0,-1,0)));
    // setAtomsDerivatives (valueb,1,matmul(getPbc().getInvBox(),Vector(0,+1,0)));
    //  valueb->set(Tools::pbc(d[1]));
    // setAtomsDerivatives (valuec,0,matmul(getPbc().getInvBox(),Vector(0,0,-1)));
    // setAtomsDerivatives (valuec,1,matmul(getPbc().getInvBox(),Vector(0,0,+1)));
    // // calculate derivatives
    unsigned offset = refposA.size();
    double unit_vec_sum = unit_vec[0] + unit_vec[1] + unit_vec[2];
    std::vector <Vector> identity = {Vector(1.0, 0.0, 0.0),
                                     Vector(0.0, 1.0, 0.0),
                                     Vector(0.0, 0.0, 1.0)};

    for (unsigned ia=0; ia < currposB.size(); ia++) {
        for (unsigned p=0; p<3; p++) {

            //fprintf(stderr, "%g %g %g\n", rot.dQ0_2[ia][p][0],rot.dQ0_2[ia][p][0], rot.dQ0_2[ia][p][0]);
            Vector deriv = rot.quaternionRotate(rot.quaternionInvert(rot.q), unit_vec_sum * unit_vec[p] * identity[p]);
            // ToDo add each atom weight
            setAtomsDerivatives(distptr[p], ia+offset, deriv * 1/currposB.size());
            //sum_grad += rot.dQ0_2[ia][p];
        }
        //fprintf(stderr, "%g %g %g\n", sum_grad[0],sum_grad[1], sum_grad[2]);
    }
    //fprintf(stderr, "Frmae ************************************\n");
    // calculate the fitting group  derivatives
    //rotinv = rot.inverse();
    // add the center of geometry contribution to the gradients
    // https://github.com/Colvars/colvars/blob/914a4eee106cb84506bd87f72ad38390732dd8a2/src/colvaratoms.cpp#L1211
    if (!NoFitGradients){
        Vector atom_grad;
        for (unsigned i=0; i < currposB.size(); i++){
            //pos_orig = rot_inv.rotate(rot_pos);
            atom_grad.zero();
            for (unsigned p=0; p<3; p++)
                atom_grad += (rot.quaternionRotate(rot.quaternionInvert(rot.q), unit_vec_sum * unit_vec[p] * identity[p]));

            Vector4d dxdq = rot.position_derivative_inner(translate_posB[i], atom_grad);

            for (unsigned j=0; j < refposA.size(); j++) {
                for (unsigned p=0; p<3; p++) {

                    //fprintf(stderr, "%g %g %g\n", rot1.dQ0_2[ia][p][0],rot1.dQ0_2[ia][p][0], rot1.dQ0_2[ia][p][0]);
                    setAtomsDerivatives(distptr[p], j, dxdq[p] * unit_vec_sum * -1 * unit_vec[p] * identity[p] * 1/currposA.size());
                }

            }
        }
    }



    //Now that all derivatives have been calculated set the system box derivatives.
    for (unsigned i=0;i<3;i++)
        setBoxDerivativesNoPbc(distptr[i]);

}

// calculator
void DistanceDir::calculate(){
    //makeWhole();
    #ifdef DEBUG__CHENP
    fprintf (stderr, "= = Debug: DistanceDir::calculate has been called.\n");
    #endif

    // Obtain current position, reference position,
    optimalAlignment1( getPositions() );

}

}
}
