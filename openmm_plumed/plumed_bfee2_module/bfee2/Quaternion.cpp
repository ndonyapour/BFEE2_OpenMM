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

//+PLUMEDOC COLVAR QUATERNION
/*
 * Calculates quaternion rotation to a reference.
 *
 * Version: 0.2 - Poker Chen 07.01.2016
 * Added relative quaternion between two domains.
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
 * If both REFERENCE and REFERENCE_B are given, then the relative orientation
 * between the two groups are given is calculated, using the convention
 *  qB = qr.qA, or, qr=qB.qA^-1 = qB.(-qA)
 *
 * NORM_DIRECTION can also be set to remove the 2-fold degeneracy of q==-q,
 * When using restraints, this should usually be set to the restraint quaternion itself.
 * PLUMED will automatically normalise this, so you can use (1,1,1,1) -> (0.5,0.5,0.5,0.5)
 *
\par Examples

q: QUATERNION REFERENCE=./refA.pdb REFERENCE_B=./refB.pdb NORM_DIRECTION=1,1,0,0

RESTRAINT ARG=q.w,q[0],q[1],q[2] AT=0.70710678118,0.70710678118,0,0 KAPPA=24.94,24.94,24.94,24.94 LABEL=restx

*/
//+ENDPLUMEDOC

//Declare class and content variables
class Quaternion : public Colvar {
    private:
    unsigned int refnat1;
    std::vector<AtomNumber> refind1;
    std::vector<Vector> refpos1;
    std::vector<double> refw1;
    std::vector<double> refdisp1;
    Vector refc1;
    bool bRefc1_is_calculated;
    bool bRef1_is_centered;
    bool NoFitGradients;

    //PLMD::RMSDBase* rmsd2;
    unsigned int refnat2;
    std::vector<AtomNumber> refind2;
    std::vector<Vector> refpos2;
    std::vector<double> refw2;
    std::vector<double> refdisp2;
    Vector refc2;
    bool bRefc2_is_calculated;
    bool bRef2_is_centered;
    Matrix<double> S12;
    Vector4d quat2;
    double lambda12;

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
  explicit Quaternion(const ActionOptions&);
  ~Quaternion();
  void clear();
  // set reference coordinates, remove the com by using uniform weights
  void setReferenceA(const std::vector<AtomNumber> & inpind, const std::vector<Vector> & inppos,
                     const std::vector<double> & inpw, const std::vector<double> & inpdisp );
  void setReferenceB(const std::vector<AtomNumber> & inpind, const std::vector<Vector> & inppos,
                     const std::vector<double> & inpw, const std::vector<double> & inpdisp );
  //void setReference2(const std::vector<Vector> & inppos, const std::vector<double> & inpw, const std::vector<double> & inpdisp );
  void setReferenceA(const PDB inppdb );
  void setReferenceB(const PDB inppdb );
  std::vector<Value*> setupQuaternionColvarPtr(Vector4d q);

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
PLUMED_REGISTER_ACTION(Quaternion,"QUATERNION")

void Quaternion::registerKeywords(Keywords& keys){
  Colvar::registerKeywords(keys);
  keys.add("compulsory","REFERENCE","A file in pdb format containing the reference structure and the atoms involved in the CV.");
  keys.add("optional","REFERENCE_B","A second reference file for the second group to calculate relative orientation.");
  keys.add("compulsory","NORM_DIRECTION","w","q-space is double defined such that q=-q, so it is conventional to define an alignment direction."
                      "This defaults to (1,0,0,0) in general literature, but should be assigned to a similar direction to a restraint."
                      "Options: x=(0,1,0,0), y, z, or an arbitrary quaternion.");

  keys.addFlag("NoFitGradients", false, "Ignores the fiiting rotational gradients contribution to colvar gradients");
  keys.remove("NOPBC");
  keys.addFlag("COMPONENTS", true, "(Compulsary) Calculate the quaternion as 4-individual colvars, stored as label.w, label[0], label[1] and label[2]");
  keys.addOutputComponent("w","COMPONENTS","the w-component of the rotational quaternion");
  keys.addOutputComponent("x","COMPONENTS","the x-component of the rotational quaternion");
  keys.addOutputComponent("y","COMPONENTS","the y-component of the rotational quaternion");
  keys.addOutputComponent("z","COMPONENTS","the z-component of the rotational quaternion");
}

Vector4d Quaternion::stringToQuat(const string str, const char* fs)
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
    if (i!=4) fprintf(stderr,"= = = WARNING: did not read 4 components to the normalisation quaternion!\n");
    //fprintf (stderr, "= = Debug: finished STRINGTOQUAT..");
    //fprintf(stderr,"q: %g %g %g %g\n", q[0], q[1], q[2], q[3]);
    return q;
}

//Constructor
Quaternion::Quaternion(const ActionOptions&ao):
//PLUMED_COLVAR_INIT(ao)
//Initialise posre1 to hold 1 position and 0 derivatives.
//Initialise mypack to be empty with no arguments and no atoms.
//PLUMED_COLVAR_INIT(ao),posder1(1,0), mypack1(0,0,posder1), posder2(1,0)
PLUMED_COLVAR_INIT(ao)
{
    fprintf(stderr, "= = Debug: Constructing the QUATERNION colvar...\n");
    bRefc1_is_calculated = false;
    bRef1_is_centered = false;
    bRefc2_is_calculated = false;
    bRef2_is_centered = false;

    string reffile1;
    string reffile2;

    parse("REFERENCE", reffile1);
    parse("REFERENCE_B", reffile2);
    string type;
    type.assign("OPTIMAL");
    string normdir;
    parse("NORM_DIRECTION", normdir);
    parseFlag("NoFitGradients", NoFitGradients);

    //Setup the normalisation direction of q. Initialise a vector
    //and then give its address to the COLVAR.
    Vector4d nq(1.0, 0.0 ,0.0 , 0.0);
    fprintf (stderr, "= = = Debug: normdir argument: %s, comparisons %i %i %i %i\n",
            normdir.c_str(),
            normdir.compare("w"),normdir.compare("x"),normdir.compare("y"),normdir.compare("z"));
    if ( normdir.compare("w")==0 ) {Vector4d dq(1.0, 0.0, 0.0, 0.0); nq=dq;}
    else if ( normdir.compare("x")==0 ) {Vector4d dq(0.0, 1.0, 0.0, 0.0); nq=dq;}
    else if ( normdir.compare("y")==0 ) {Vector4d dq(0.0, 0.0, 1.0, 0.0); nq=dq;}
    else if ( normdir.compare("z")==0 ) {Vector4d dq(0.0, 0.0, 0.0, 1.0); nq=dq;}
    else {
        fprintf (stderr, "= = = Debug: Detected custom q input: %s\n", normdir.c_str());
        Vector4d dq = stringToQuat(normdir, ",");
        fprintf (stderr, "= = = Debug: Parsed qvals: %g %g %g %g\n",
                dq[0],dq[1],dq[2],dq[3]);
        //parseVector("NORM_DIRECTION",qvals);

        dq /= dq.modulo(); //Normalise for safety & flexibility.
        nq = dq;
    }
    normquat = nq;
    fprintf (stderr, "= = = Debug: normalisation-q: %g %g %g %g\n", nq[0], nq[1], nq[2], nq[3]);
    Vector4d qq; qq.zero();
    quat = qq;
    checkRead();

    fprintf (stderr, "= = = Debug: Will now add components...\n");
    addComponentWithDerivatives("w"); componentIsNotPeriodic("w");
    addComponentWithDerivatives("x"); componentIsNotPeriodic("x");
    addComponentWithDerivatives("y"); componentIsNotPeriodic("y");
    addComponentWithDerivatives("z"); componentIsNotPeriodic("z");
    log<<"NOTE: q is stored as four components (w x y z).\n";

    PDB pdbA; //PDB storage of reference coordinates 1
    PDB pdbB; //PDB storage of reference coordinates 2

    fprintf (stderr, "= = = Debug: Will now read REFERENCE pdb file...\n");
    // read everything in ang and transform to nm if we are not in natural units
    if( !pdbA.read(reffile1,plumed.getAtoms().usingNaturalUnits(),0.1/atoms.getUnits().getLength()) )
        error("missing input file " + reffile1 );

    //Store reference positions with zeroed barycenters.
    setReferenceA( pdbA );
    fprintf(stderr, "= = = = Debug: read and parse finished.\n");

    if ( !reffile2.empty() ) {
        fprintf (stderr, "= = = Debug: REFERENCE_B found, parsing second pdb file...\n");
        if( !pdbB.read(reffile2,plumed.getAtoms().usingNaturalUnits(),0.1/atoms.getUnits().getLength()) )
            error("missing input file " + reffile2 );

        //Store reference positions with zeroed barycenters.
        setReferenceB( pdbB );
        fprintf (stderr, "= = = = Debug: read and parse finished ...\n");
    }

    // The colvar module takes in only 1 set of atoms.
    // Therefore, will need additional code to handle relative orientation from two diomains.
    fprintf (stderr, "= = = Debug: Requesting atoms for tracking...\n");
    if (!bRefc2_is_calculated) {
        // Simple case of 1 reference, absolute quaternion.
        std::vector<AtomNumber> atoms;
        atoms = refind1;
        //rmsd1 = metricRegister().create<RMSDBase>(type,pdb1);
        //rmsd1->getAtomRequests( atoms );
        fprintf(stderr,"= = = = Debug: First 3 entries of pdbA atoms: %u %u %u\n",
                atoms[0].index(), atoms[1].index(), atoms[2].index());

        requestAtoms( atoms );
        log.printf("  reference from file %s\n",reffile1.c_str());
        log.printf("  which contains %d atoms\n",getNumberOfAtoms());
    } else {
    //if (bRefc2_is_calculated) {
        // 2 Reference files, relative quaternions.
        std::vector<AtomNumber> atoms;
        fprintf(stderr,"= = = = Debug: First 3 entries of %lu pdbA atoms: %u %u %u\n",
                refind1.size(), refind1[0].index(), refind1[1].index(), refind1[2].index());
        fprintf(stderr,"= = = = Debug: First 3 entries of %lu pdbB atoms: %u %u %u\n",
                refind2.size(), refind2[0].index(), refind2[1].index(), refind2[2].index());

        atoms.reserve( refind1.size()+refind2.size() );
        atoms.insert( atoms.end(), refind1.begin(), refind1.end());
        atoms.insert( atoms.end(), refind2.begin(), refind2.end());

        requestAtoms( atoms );
        log.printf("  reference from files %s and %s\n",
                reffile1.c_str(), reffile2.c_str());
        log.printf("  which contains %d atoms\n",getNumberOfAtoms());
    }
    fprintf (stderr, "= = = = Debug: Request finished.\n");
     fprintf (stderr, "= = = Debug: finished constructing QUATERNION.\n");
    //atoms.clearFullList();
}

Quaternion::~Quaternion(){
  //delete rmsd1;
  //if (rmsd2) delete rmsd2;
}

void Quaternion::clear(){
    refind1.clear();
    refpos1.clear();
    refind2.clear();
    refpos2.clear();
}

//Currently uses a C-like approach, should encapsulate into an object.
//Prefilters the atom list to remove all atoms with zero weight, reducing the passing of
//data between plumed and the MD engine.

void Quaternion::setReferenceA(const std::vector<AtomNumber> & inpind, const std::vector<Vector> & inppos,
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
    refpos1.reserve(refnat1);
    refw1.reserve(refnat1);
    refdisp1.reserve(refnat1);
    for (unsigned i=0; i<refnat1; i++)
    {
        //fprintf(stderr,"= = = DEBUG check for atom indices: %i ... %i\n", refind1[i].index(), i);
        refpos1.push_back(inppos[atom_internal_idxs[i]]);
        refw1.push_back(inpw[atom_internal_idxs[i]]);
        refdisp1.push_back(inpdisp[atom_internal_idxs[i]]);
        //refind1[i]=inpind[i];
        //fprintf(stderr,"atom idxs... %u\n", atom_internal_idxs[i]);
    }

    // Only need to iterate over subindices.
    // we might need to change this
    double wtot=0.0, disptot=0.0;  //refw1[i]*
    for(unsigned i=0;i<refnat1;i++) {refc1+=refpos1[i]; wtot+=refw1[i]; disptot+=refdisp1[i];}
    fprintf(stderr, "= = = = = Debug setReferenceA(): wtot is %g and disptot is %g\n", wtot, disptot);
    refc1/=refpos1.size();
    for(unsigned i=0;i<refnat1;i++) {refpos1[i]-=refc1; refw1[i]=refw1[i]/wtot ; refdisp1[i]=refdisp1[i]/disptot; }
    bRefc1_is_calculated=true;
    bRef1_is_centered=true;
}
void Quaternion::setReferenceA(const PDB inppdb ){
    setReferenceA( inppdb.getAtomNumbers(), inppdb.getPositions(), inppdb.getOccupancy(), inppdb.getBeta() );
}

void Quaternion::setReferenceB(const std::vector<AtomNumber> & inpind, const std::vector<Vector> & inppos,
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
    refpos2.reserve(refnat2);
    refw2.reserve(refnat2);
    refdisp2.reserve(refnat2);
    for (unsigned i=0;i<refnat2;i++)
    {
        //fprintf(stderr,"= = = DEBUG check for atom indices: %i ... %i\n", refind2[i].index(), i);
        refpos2.push_back(inppos[atom_internal_idxs[i]]);
        refw2.push_back(inpw[atom_internal_idxs[i]]);
        refdisp2.push_back(inpdisp[atom_internal_idxs[i]]);
        //refind2[i]=inpind[i];
        //fprintf(stderr,"... %i\n", refind2[i].index());
    }

    // Only need to iterate over subindices.
    double wtot=0.0, disptot=0.0;
    for(unsigned i=0;i<refnat2;i++) {refc2+=refpos2[i]; wtot+=refw2[i]; disptot+=refdisp2[i];}
    fprintf(stderr, "= = = = = Debug setReferenceB(): wtot is %g and disptot is %g\n", wtot, disptot);
    refc2/=refpos2.size();
    for(unsigned i=0;i<refnat2;i++) {refpos2[i]-=refc2; refw2[i]=refw2[i]/wtot ; refdisp2[i]=refdisp2[i]/disptot; }
    bRefc2_is_calculated=true;
    bRef2_is_centered=true;
}

void Quaternion::setReferenceB(const PDB inppdb ){
    setReferenceB( inppdb.getAtomNumbers(), inppdb.getPositions(), inppdb.getOccupancy(), inppdb.getBeta() );
}

Vector Quaternion::calculateCOG(const std::vector<Vector> pos)
{
    Vector cog = Vector(0,0,0);
    for (unsigned i=0; i<pos.size(); i++)
            cog += pos[i];
    cog /= pos.size();
    return cog;
}

std::vector<Vector> Quaternion::translateCoordinates(const std::vector<Vector> pos, const Vector t)
{
    std::vector<Vector> translated_pos;
    for (unsigned i=0; i<pos.size(); i++) {
        translated_pos.push_back(pos[i] - t);
    }
 return translated_pos;
}


std::vector<Value*> Quaternion::setupQuaternionColvarPtr(Vector4d q)
{
    Value* valuew=getPntrToComponent("w");
    Value* valuex=getPntrToComponent("x");
    Value* valuey=getPntrToComponent("y");
    Value* valuez=getPntrToComponent("z");
    std::vector<Value*> qptr;

    qptr.push_back(valuew); qptr.push_back(valuex);
    qptr.push_back(valuey); qptr.push_back(valuez);

    for (unsigned i=0;i<4;i++) qptr[i]->set(q[i]);

    return qptr;
}


void Quaternion::optimalAlignment1(const std::vector<Vector> & currpos)
{
    /* General procedure to calculate the quaternion value is to provide first a best fit.
    * Reference notation in NAMD, PLUMED, and Dill.
    * NOTES: Copy over RMSDCoreData::doCoreCalc first, and then do relevant pruning.
    * 1) Make the barycenters zero.
    * 2) gather the second moment/correlation matrix C = sum_1^N X1 X2T - is curly R in Ken Dill (2004), eq. 5
    * 3) Create the equivalent quaternion matrix/overlap matrix S  - is curly F, eq. 10
    * 4) Diagonalise this 4x4 matrix, the maximum eigenvector is almost always
    *    the best rotation, the first eigenvector by convention.
    * 5) Can use for both RMSD and rotation matrices.
    * 6) Finally calculate the atomic derivatives with respect to q
    *      Sij = sum( x_k*xref_k ), so if i=j, dSdx_j = xref_j ....
    */
    //fprintf (stderr, "= = Debug: Starting optimalAlignment1...\n");

    /* When a second reference is given:
     * 0) Assume that two reference frames are in the same orientation (1,0,0,0).
     * 1) rotate the second reference PDB according to q1,
     * 2) calculate the quaternion value to this new frame. This is q1->2.
     * 3) Then apply forces.
     */



    std::vector<Vector> translate_pos1, currpos1, currpos2 ;



    //currpos1.resize(refpos1.size(), Vector);
    for (unsigned i=0; i<refpos1.size(); i++ )
        currpos1.push_back(currpos[i]);

    for (unsigned i=0; i<refpos2.size(); i++ )
        currpos2.push_back(currpos[i+refpos1.size()]);
    //fprintf(stderr, "refpos1 ******************* %i\n", currpos1.size());



    std::vector<Value*> qptr;
    if (!bRefc2_is_calculated) {
        qrotation rot;
        Vector currpos_cog = calculateCOG(currpos1);
        translate_pos1 = translateCoordinates(currpos1, currpos_cog);
        rot.request_group2_gradients(currpos.size());
        rot.calc_optimal_rotation(refpos1, translate_pos1, normquat);


        qptr = setupQuaternionColvarPtr(rot.q);
        // gradeints of dq/dx
        for (unsigned ia=0; ia<refpos1.size(); ia++) {
            for (unsigned p=0; p<4; p++) {

                    //fprintf(stderr, "%g %g %g\n", rot1.dQ0_2[ia][p][0],rot1.dQ0_2[ia][p][0], rot1.dQ0_2[ia][p][0]);
                    // ToDo add each atom weight
                    setAtomsDerivatives(qptr[p], ia, rot.dQ0_2[ia][p] * 1/refpos1.size());
             }
        }
    }

    else {
        // when we have two groups

        // Rotation of the first group (fitting group) Center both to the center fitting group
        std::vector<Vector>  translate_fit, translate_pos;
        qrotation rotfit, rot;
        Vector fit_cog = calculateCOG(currpos1);

        translate_fit = translateCoordinates(currpos1, fit_cog);

        //Vector pos_cog = calculateCOG(currpos2);
        translate_pos = translateCoordinates(currpos2, fit_cog);

        // You need to request gradients first
        rotfit.request_group1_gradients(currpos1.size());
        rotfit.calc_optimal_rotation(translate_fit, refpos1, normquat);

        // apply fitting rotation
        std::vector<Vector> rot_fit, rot_pos;
        rot_fit = rotfit.rotateCoordinates(rotfit.q, translate_fit);
        rot_pos = rotfit.rotateCoordinates(rotfit.q, translate_pos);

        // main rotation
        rot.request_group2_gradients(currpos2.size());
        rot.calc_optimal_rotation(refpos2, rot_pos, normquat);

        // set the q values
        qptr = setupQuaternionColvarPtr(rot.q);


        // calculate derivatives
        unsigned offset = refpos1.size();
        for (unsigned ia=0; ia < currpos2.size(); ia++) {
            //Vector sum_grad(0.0, 0.0, 0.0);
            for (unsigned p=0; p<4; p++) {

                //fprintf(stderr, "%g %g %g\n", rot.dQ0_2[ia][p][0],rot.dQ0_2[ia][p][0], rot.dQ0_2[ia][p][0]);
                // rotate back the dq/dx vector
                Vector deriv = rot.quaternionRotate(rot.quaternionInvert(rotfit.q), rot.dQ0_2[ia][p]);
                // ToDo add each atom weight
                setAtomsDerivatives(qptr[p], ia+offset, deriv * 1/currpos2.size());
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
            for (unsigned i=0; i < currpos2.size(); i++){
                //pos_orig = rot_inv.rotate(rot_pos);
                atom_grad.zero();
                for (unsigned p=0; p<4; p++)
                    // gradients of second group
                    atom_grad += (rot.quaternionRotate(rot.quaternionInvert(rotfit.q), rot.dQ0_2[i][p]) * 1/currpos2.size());

                Vector4d dxdq = rotfit.position_derivative_inner(translate_pos[i], atom_grad);

                for (unsigned j=0; j < refpos1.size(); j++) {
                    for (unsigned p=0; p<4; p++) {

                        //fprintf(stderr, "%g %g %g\n", rot1.dQ0_2[ia][p][0],rot1.dQ0_2[ia][p][0], rot1.dQ0_2[ia][p][0]);
                        setAtomsDerivatives(qptr[p], j, dxdq[p]*rotfit.dQ0_1[j][p]*1/currpos1.size());
                    }

                }
            }
        }

    }

    //Now that all derivatives have been calculated set the system box derivatives.
    for (unsigned i=0;i<4;i++)
        setBoxDerivativesNoPbc(qptr[i]);

    #ifdef DEBUG__CHENP
    fprintf (stderr, "= = = Debug: Finished optimalAlignment1. q = (%g %g %g %g)\n",
                     qmat[0][0], qmat[0][1], qmat[0][2], qmat[0][3]);
    #endif
    return;
}

// calculator
void Quaternion::calculate(){
    //makeWhole();
    #ifdef DEBUG__CHENP
    fprintf (stderr, "= = Debug: QUATERNION::calculate has been called.\n");
    #endif

    // Obtain current position, reference position,
    optimalAlignment1( getPositions() );

}

}
}
