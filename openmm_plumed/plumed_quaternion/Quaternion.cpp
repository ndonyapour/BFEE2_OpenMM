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
#include "Colvar.h"
#include "core/PlumedMain.h"
#include "ActionRegister.h"
#include "tools/PDB.h"
#include "reference/RMSDBase.h"
#include "reference/MetricRegister.h"
#include "core/Atoms.h"

#include <string>
#include <sstream> //For parsing the norm quaternion.
#include <cmath>

//#define DEBUG__CHENP

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

RESTRAINT ARG=q.w,q.x,q.y,q.z AT=0.70710678118,0.70710678118,0,0 KAPPA=24.94,24.94,24.94,24.94 LABEL=restx

*/
//+ENDPLUMEDOC

//Declare class and content variables
class Quaternion : public Colvar {
    private:
    //bool pbc; No PBC.
    //MultiValue posder1; //a flexible vector to store positions and derivatives.
    //MultiValue posder2; //a flexible vector to store positions and derivatives.
    //ReferenceValuePack mypack1; //Object to carry derivatives from calculations to system.
    //ReferenceValuePack mypack2;
    //PLMD::RMSDBase* rmsd1;
    unsigned int refnat1;
    std::vector<AtomNumber> refind1;
    std::vector<Vector> refpos1;
    std::vector<double> refw1;
    std::vector<double> refdisp1;
    Vector refc1;
    bool bRefc1_is_calculated;
    bool bRef1_is_centered;
    Matrix<double> S01;
    Vector4d quat1;
    double lambda01;

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

// Obtaining data
  Vector4d getQfromEigenvecs(void);
  double   getLfromEigenvals(void);
  std::vector<Value*> setupQuaternionColvarPtr(Vector4d q);

// active methods:
  //Because I don't have the time to fuck around with parseVector from Bias.
  Vector4d stringToQuat(const string str, const char *fs);

  std::vector<Vector> rotateCoordinates(const Vector4d q, const std::vector<Vector> pos, bool bDoCenter);

  void calculateSmat(const unsigned nvals,
        const std::vector<Vector> ref, const std::vector<double> w,
        const unsigned offset, const std::vector<Vector> & loc);
  void diagMatrix(void);

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
  //This must always be optimal rotations.
  //keys.add("compulsory","TYPE","SIMPLE","the manner in which RMSD alignment is performed.  Should be OPTIMAL or SIMPLE.");
  //keys.addFlag("SQUARED",false," This should be setted if you want MSD instead of RMSD ");
  keys.remove("NOPBC");
  keys.addFlag("COMPONENTS",true,"(Compulsary) Calculate the quaternion as 4-individual colvars, stored as label.w, label.x, label.y and label.z");
  keys.addOutputComponent("w","COMPONENTS","the w-component of the rotational quaternion");
  keys.addOutputComponent("x","COMPONENTS","the x-component of the rotational quaternion");
  keys.addOutputComponent("y","COMPONENTS","the y-component of the rotational quaternion");
  keys.addOutputComponent("z","COMPONENTS","the z-component of the rotational quaternion");
  //keys.addFlag("TEMPLATE_DEFAULT_OFF_FLAG",false,"flags that are by default not performed should be specified like this");
  //keys.addFlag("TEMPLATE_DEFAULT_ON_FLAG",true,"flags that are by default performed should be specified like this");
  //keys.add("compulsory","TEMPLATE_COMPULSORY","all compulsory keywords should be added like this with a description here");
  //keys.add("optional","TEMPLATE_OPTIONAL","all optional keywords that have input should be added like a description here");
  //keys.add("atoms","TEMPLATE_INPUT","the keyword with which you specify what atoms to use should be added like this");
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
PLUMED_COLVAR_INIT(ao),
eigenvals(4,0.0)
{
    fprintf (stderr, "= = Debug: Constructing the QUATERNION colvar...\n");
    bRefc1_is_calculated = false;
    bRef1_is_centered = false;
    bRefc2_is_calculated = false;
    bRef2_is_centered = false;

    Smat = Matrix<double>(4,4);
    qmat = Matrix<double>(4,4);
    S01  = Matrix<double>(4,4);
    S12  = Matrix<double>(4,4);
    quat1= Vector4d(0.0,0.0,0.0,0.0);
    quat2= Vector4d(0.0,0.0,0.0,0.0);

    string reffile1;
    string reffile2;

    parse("REFERENCE",reffile1);
    parse("REFERENCE_B", reffile2);
    string type;
    type.assign("OPTIMAL");
    string normdir;
    parse("NORM_DIRECTION", normdir);
    //parse("TYPE",type);

    //Setup the normalisation direction of q. Initialise a vector
    //and then give its address to the COLVAR.
    Vector4d nq(1.0,0.0,0.0,0.0);
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

        //rmsd1 = metricRegister().create<RMSDBase>(type,pdb1);
        //rmsd1->getAtomRequests( temp1 );
        //rmsd2 = metricRegister().create<RMSDBase>(type,pdb2);
        //rmsd2->getAtomRequests( temp2 );
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
    // Setup the derivative pack
    //posder1.resize( 1, 3*atoms.size()+9 ); mypack1.resize( 0, atoms.size() );
    //for(unsigned i=0;i<atoms.size();++i) mypack1.setAtomIndex( i, i );


  /*if ( !reference2.empty() ) {
    if ( !pdb2.read(reference2,plumed.getAtoms().usingNaturalUnits(),0.1/atoms.getUnits().getLength()) )
      error("missing input file " + reference2 );

    rmsd2 = metricRegister().create<RMSDBase>(type,pdb2);

    rmsd2->getAtomRequests( atoms );
    //  rmsd->setNumberOfAtoms( atoms.size() );
    requestAtoms( atoms );
    // Setup the derivative pack
    posder2.resize( 1, 3*atoms.size()+9 ); mypack2.resize( 0, atoms.size() );
    for(unsigned i=0;i<atoms.size();++i) mypack2.setAtomIndex( i, i );

    log.printf("  reference from file %s\n",reference2.c_str());
    log.printf("  which contains %d atoms\n",getNumberOfAtoms());
  }*/
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

    //Preallocate to maximum possible
    //refind1.reserve(nvals);

    //Filter out the zero weight data.
    // fprintf(stderr, "inpos size=%u inpw=%u\n", inppos.size(), inpw.size());
    // for(unsigned i=0; i<inpind.size(); ++i)
    //     fprintf(stderr, "index=%u, serial=%u \n", inpind[i].index(), inpind[i].serial());

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
    double wtot=0.0, disptot=0.0;
    for(unsigned i=0;i<refnat1;i++) {refc1+=refw1[i]*refpos1[i]; wtot+=refw1[i]; disptot+=refdisp1[i];}
    fprintf(stderr, "= = = = = Debug setReferenceA(): wtot is %g and disptot is %g\n", wtot, disptot);
    refc1/=wtot;
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
    for(unsigned i=0;i<refnat2;i++) {refc2+=refw2[i]*refpos2[i]; wtot+=refw2[i]; disptot+=refdisp2[i];}
    fprintf(stderr, "= = = = = Debug setReferenceB(): wtot is %g and disptot is %g\n", wtot, disptot);
    refc2/=wtot;
    for(unsigned i=0;i<refnat2;i++) {refpos2[i]-=refc2; refw2[i]=refw2[i]/wtot ; refdisp2[i]=refdisp2[i]/disptot; }
    bRefc2_is_calculated=true;
    bRef2_is_centered=true;
}

void Quaternion::setReferenceB(const PDB inppdb ){
    setReferenceB( inppdb.getAtomNumbers(), inppdb.getPositions(), inppdb.getOccupancy(), inppdb.getBeta() );
}

//void Quaternion::setReference2(const std::vector<Vector> & inppos,
//                               const std::vector<double> & inpw, const std::vector<double> & inpdisp ){
//  unsigned nvals=inppos.size();
//  refpos2=inppos;
//  plumed_massert(refw2.empty(),"you should first clear() an RMSD object, then set a new reference");
//  plumed_massert(refdisp2.empty(),"you should first clear() an RMSD object, then set a new reference");
//  refw2.resize(nvals,0.0);
//  refdisp2.resize(nvals,0.0);
//  double wtot=0, disptot=0;
//  for(unsigned i=0;i<nvals;i++) {refc2+=inpw[i]*refpos2[i]; wtot+=inpw[i]; disptot+=inpdisp[i];}
//  refc2/=wtot;
//  for(unsigned i=0;i<nvals;i++) {refpos2[i]-=refc2; refw2[i]=inpw[i]/wtot ; refdisp2[i]=inpdisp[i]/disptot; }
//  bRefc2_is_calculated=true;
//  bRef2_is_centered=true;
//}
//void Quaternion::setReference2(const PDB inppdb ){
//    setReference2( inppdb.getPositions(), inppdb.getOccupancy(), inppdb.getBeta() );
//}

//Returns a copy of coordinates pos, rotated according to q.
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

//Returns a copy of coordinates pos, rotated according to q.
std::vector<Vector> Quaternion::rotateCoordinates(const Vector4d q, const std::vector<Vector> pos, bool bDoCenter)
{
    std::vector<Vector> rot;
    #ifdef DEBUG__CHENP
    fprintf(stderr,"= = = DEBUG: ROTATECOORDINATES has been called using q(%g %g %g %g)\n",
            q[0], q[1], q[2], q[3]);
    if (bDoCenter) fprintf(stderr,"         NB: with centering..\n");
    fprintf(stderr,"Positions of first three atoms:\n"
            " ( %g %g %g ) ( %g %g %g ) ( %g %g %g )\n",
            pos[0][0],pos[0][1],pos[0][2],pos[1][0],pos[1][1],pos[1][2],pos[2][0],pos[2][1],pos[2][2]);
    #endif
    unsigned ntot = pos.size();
    rot.resize(ntot,Vector(0,0,0));

    if (bDoCenter)
    {
        Vector posc = Vector(0,0,0);
        for (unsigned i=0;i<ntot;i++)
            posc+=pos[i];
        posc/=ntot;
        for (unsigned i=0;i<ntot;i++)
            rot[i]=quaternionRotate(q,pos[i]-posc);
    } else {
        for (unsigned i=0;i<ntot;i++)
            rot[i]=quaternionRotate(q, pos[i]);
    }

    #ifdef DEBUG__CHENP
    fprintf(stderr,"= = = DEBUG: finished ROTATECOORDINATES. \n");
    fprintf(stderr,"Rotated Positions of first three atoms:\n"
            " ( %g %g %g ) ( %g %g %g ) ( %g %g %g )\n",
            rot[0][0],rot[0][1],rot[0][2],rot[1][0],rot[1][1],rot[1][2],rot[2][0],rot[2][1],rot[2][2]);
    #endif
    return rot;
}

Vector4d Quaternion::getQfromEigenvecs(void)
{
    return Vector4d(qmat[0][0],qmat[0][1],qmat[0][2],qmat[0][3]);
}

double   Quaternion::getLfromEigenvals(void)
{
    return eigenvals[0];
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

// Generic quaternion calculator. Able to function on submatrices
// Using formalism q being the rotation from reference to the local frame.
// Note that this is the negative of the NAMD overlap matrix (it doesn't really matter).
void Quaternion::calculateSmat(const unsigned nvals,
        const std::vector<Vector> ref, const std::vector<double> w,
        const unsigned offset, const std::vector<Vector> & loc_in)
{
//    std::vector<Vector3d> loc = *loc_in;
//    Matrix<double> S = *S_out;
//    std::vector<double> l= *l_out;
//    Matrix<double> q = *q_out;

    Vector posc = Vector(0.0,0.0,0.0);
    double wtot; wtot=0.0;
    std::vector<Vector> loc = loc_in;
    #ifdef DEBUG__CHENP
    fprintf (stderr, "= = = Debug: CALCULATE_SMAT called\n");
    #endif
    unsigned ntot=nvals+offset;
    if (ntot>loc.size())
    {
        std::string msg="CALCULATE_SMAT FAILED: OFFSET + NVALS EXCEEDS CURRENT POSITION SIZE";
        plumed_merror(msg);
    }

    //Auto-center local coordinates
    //fprintf (stderr, "= = = = Debug: position 0: %g %g %g\n", loc[0][0],loc[0][1],loc[0][2]);
    for(unsigned i=0;i<nvals;i++) {posc += w[i]*loc[i+offset]; wtot += w[i];}
    //fprintf(stderr,"%g - %g %g %g\n", wtot, posc[0],posc[1],posc[2] );
    posc /= wtot;
    for(unsigned j=offset;j<ntot;j++) {loc[j]-=posc;}
    //fprintf (stderr, "= = = = Debug: position 0: %g %g %g\n", loc[0][0],loc[0][1],loc[0][2]);
    #ifdef DEBUG__CHENP
    fprintf (stderr, "= = = = Debug: CALCULATE_SMAT: Current frame has been centered: %g %g %g\n",
                     posc[0], posc[1], posc[2]);
    #endif
    // (2)
    // second expensive loop: compute second moments wrt centers
    // Already subtracted from the data.
    //Vector cp; cp.zero(); //if(!cpositions_is_removed)cp=cpositions;
    //Vector cr; cr.zero(); //if(!creference_is_removed)cr=creference;
    rr00=0.0; rr11=0.0; rr01.zero();
    unsigned iloc;
    for(unsigned iat=0;iat<nvals;iat++){
        iloc=iat+offset;
        rr00+=dotProduct(loc[iloc],loc[iloc])*w[iat];
        rr11+=dotProduct(ref[iat],ref[iat])*w[iat];
        rr01+=Tensor(ref[iat],loc[iloc])*w[iat]; //Here is the order dependence!!
    }


    // (3)
    // the quaternion matrix: this is internal
    //Matrix<double> Smat = Matrix<double>(4,4);
    //S=Sint;
    Smat[0][0]=2.0*(-rr01[0][0]-rr01[1][1]-rr01[2][2]);
    Smat[1][1]=2.0*(-rr01[0][0]+rr01[1][1]+rr01[2][2]);
    Smat[2][2]=2.0*(+rr01[0][0]-rr01[1][1]+rr01[2][2]);
    Smat[3][3]=2.0*(+rr01[0][0]+rr01[1][1]-rr01[2][2]);
    Smat[0][1]=2.0*(-rr01[1][2]+rr01[2][1]);
    Smat[0][2]=2.0*(+rr01[0][2]-rr01[2][0]);
    Smat[0][3]=2.0*(-rr01[0][1]+rr01[1][0]);
    Smat[1][2]=2.0*(-rr01[0][1]-rr01[1][0]);
    Smat[1][3]=2.0*(-rr01[0][2]-rr01[2][0]);
    Smat[2][3]=2.0*(-rr01[1][2]-rr01[2][1]);
    Smat[1][0] = Smat[0][1];
    Smat[2][0] = Smat[0][2];
    Smat[2][1] = Smat[1][2];
    Smat[3][0] = Smat[0][3];
    Smat[3][1] = Smat[1][3];
    Smat[3][2] = Smat[2][3];

    #ifdef DEBUG__CHENP
    fprintf (stderr, "= = = Debug: S overlap matrix calculated:\n");
    for(unsigned i=0;i<4;i++)
        fprintf (stderr, "           [ %g %g %g %g ]\n", Smat[i][0], Smat[i][1], Smat[i][2], Smat[i][3]);
    #endif
}

void Quaternion::diagMatrix(void)
{
    // (4) Diagonalisation
    //fprintf (stderr, "= = = Debug: Starting diagonalisation of matrix S...\n");

    // l is the eigenvalues, q are the eigenvectors/quaternions.
    int diagerror=diagMat(Smat, eigenvals, qmat);
    if (diagerror!=0){
        std::string sdiagerror;
        Tools::convert(diagerror,sdiagerror);
        std::string msg="DIAGONALIZATION FAILED WITH ERROR CODE "+sdiagerror;
        plumed_merror(msg);
    } else {
        #ifdef DEBUG__CHENP
        fprintf(stderr, "= = = = Debug: diagMat() successful.\n");
        fprintf(stderr, "= = = = Debug: Printing eigenvalues and eigenvectors:\n");
        fprintf(stderr, "       lambda: [ %g %g %g %g ]\n", eigenvals[0], eigenvals[1], eigenvals[2], eigenvals[3]);
        for (unsigned i=0;i<4;i++)
            fprintf(stderr, "       vec-%i: [ %g %g %g %g ]\n",
                    i, qmat[i][0], qmat[i][1], qmat[i][2], qmat[i][3]);
        #endif
    }

    double dot;
    //Normalise each eigenvector in the direction closer to norm
    for (unsigned i=0;i<4;i++) {
        dot=0.0;
        for (unsigned j=0;j<4;j++) {
            dot+=normquat[j]*qmat[i][j];
        }
        //printf(stderr,"Dot: %g q: %g %g %g %g\n",d dot, q[i][0], q[i][1], q[i][2], q[i][3]);
        if (dot < 0.0)
            for (unsigned j=0;j<4;j++)
                qmat[i][j]=-qmat[i][j];
    }
    #ifdef DEBUG__CHENP
    fprintf(stderr, "= = = Debug: finished DIAG_MATRIX.\n");
    #endif
}

// Quaternion version of Optimal rotations. Copied from RMSD section to simplify
// class issues that I haven't mastered yet.
// Should return the quaternion
// The first case deals with only one absolute quaternion calculation.
// The second case deals with a relative quaternion calculation, between two groups.
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

    /* Pseudocode for this function */
    /*
    alignDomain1();
    if (!bRefc2_is_calculated) {
        setSQLto1();
    } else {
        rotateReference2();
        alignDomain2();
        set SQLto12();
    }
    reportQuaterniontoOutput();
    calculateDerivatives(); That is...
    for (unsigned i=0;i<4;i++)
        for(unsigned j=0;j<getNumberOfAtoms();j++)
            setAtomsDerivatives(qptr[i], j, dqi/dxj );
    */

    std::vector<Vector> rotloc; //to make a rotatable copy
    std::vector<Vector> rotref1;
    //std::vector<Vector> rotref2;

    #ifdef DEBUG__CHENP
    fprintf(stderr,"= = = DEBUG: OPTIMALALIGNMENT1 has been called. \n");
    #endif
    calculateSmat(refpos1.size(), refpos1, refw1, 0, currpos);
    diagMatrix();
    #ifdef DEBUG__CHENP
    fprintf(stderr, "= = = = Debug: Printing eigenvalues and eigenvectors:\n");
    fprintf(stderr, "       lambda: [ %g %g %g %g ]\n", eigenvals[0], eigenvals[1], eigenvals[2], eigenvals[3]);
    for(unsigned i=0;i<4;i++)
        fprintf(stderr, "       vec-%i: [ %g %g %g %g ]\n",
                    i, qmat[i][0], qmat[i][1], qmat[i][2], qmat[i][3]);
    #endif

    lambda01 = getLfromEigenvals();
    quat1 = getQfromEigenvecs();


    if (!bRefc2_is_calculated) {
        S01=Smat;
        quat=quat1;
    } else {
        // Use q01^-1 to rotate the system into the frame of domain 1.
        // Slightly inefficient as we don't use domain 1 positions at the moment.
        //

        // for (unsigned i=0; i< refpos1.size(); i++)
        //     currposA.push_back(currpos[i]);

        //transloc = translateCoordinates(currpos, calculateCOG(currposA));
        rotloc = rotateCoordinates(quaternionInvert(quat1), currpos, false);
        //Calculate q12 in the frame of domain 1
        //cheat with fact that currpos is concatenated of domain 1 and domain 2.
        calculateSmat(refpos2.size(), refpos2, refw2, refpos1.size(), rotloc);
        diagMatrix();

        lambda12 = getLfromEigenvals();
        // q12 = q02 * q01 , so q02 = q12 * q1^-1.
        // We need q02 to rotate reference 1 in the force calculations.
        quat = getQfromEigenvecs();
        // for (int i=0; i <4; i++) fprintf(stderr, "******************** (%g %g %g %g)\n", quat[0],quat[1],quat[2], quat[3]);
        S12=Smat;
        //quat2 = quaternionProduct(quat, quaternionInvert(quat1));
    }

    #ifdef DEBUG__CHENP
    fprintf(stderr, "= = = = Debug: Now loading the pointers of COMPONENTS.\n");
    #endif
    // setup ptr to components as a vector, in order to manipulate outputs as a vector.
    // Also assign q-values to the COLVAR component.
    std::vector<Value*> qptr = setupQuaternionColvarPtr(quat);
    //setQuaternionColvars(qptr, quat);

    // Matrix<double> dS=Matrix<double>(4,4,3);
    //Matrix<Vector> dSdx = Matrix<Vector>(4,4,3);
    std::vector<std::vector<Vector > > dSdx;
    dSdx.resize(4, std::vector<Vector>( 4 ));

    //Calculate derivatives for atom j
    //This is ported in from NAMD code.
    #ifdef DEBUG__CHENP
    fprintf(stderr, "= = = = Debug: Now calculating dSij/dx & dq/dx over %i atoms.\n", int(refpos1.size()));
    #endif

    if (!bRefc2_is_calculated) {
        //Single domain. q is ref->loc. Follows group 2 of NAMD.
        for (unsigned j=0;j<refpos1.size();j++) {
            if (refw1[j]==0) continue; //Only apply forces to weighted atoms in the RMSD calculation.

            double const rx = refpos1[j][0];
            double const ry = refpos1[j][1];
            double const rz = refpos1[j][2];

            // dSijdx : derivative of the S matrix, w.r.t. atom x_j
            dSdx[0][0] = Vector(  rx,  ry,  rz);
            dSdx[1][0] = Vector( 0.0, -rz,  ry);
            dSdx[0][1] = dSdx[1][0];
            dSdx[2][0] = Vector(  rz, 0.0, -rx);
            dSdx[0][2] = dSdx[2][0];
            dSdx[3][0] = Vector( -ry,  rx, 0.0);
            dSdx[0][3] = dSdx[3][0];
            dSdx[1][1] = Vector(  rx, -ry, -rz);
            dSdx[2][1] = Vector(  ry,  rx, 0.0);
            dSdx[1][2] = dSdx[2][1];
            dSdx[3][1] = Vector(  rz, 0.0,  rx);
            dSdx[1][3] = dSdx[3][1];
            dSdx[2][2] = Vector( -rx,  ry, -rz);
            dSdx[3][2] = Vector( 0.0,  rz,  ry);
            dSdx[2][3] = dSdx[3][2];
            dSdx[3][3] = Vector( -rx, -ry,  rz);

            // dqi/dxj = Sum_i Sum_j q1i dSijdx q0j /(Norm) * qi...
            Vector dqidxj;
            for (unsigned i=0;i<4;i++) {
                //Calculate and append the derivatives due to each q-component separately
                dqidxj.zero();
                for (unsigned a=0;a<4;a++) {
                    for (unsigned b=0;b<4;b++) {
                        double fact=qmat[1][a]*qmat[0][b]/(eigenvals[0]-eigenvals[1])*qmat[1][i];
                              fact+=qmat[2][a]*qmat[0][b]/(eigenvals[0]-eigenvals[2])*qmat[2][i];
                              fact+=qmat[3][a]*qmat[0][b]/(eigenvals[0]-eigenvals[3])*qmat[3][i];
                        dqidxj+= -1.0*fact*dSdx[a][b];
                        // Note this is negative of the NAMD because Smat is itself the negative.
                    }
                }
                //
                setAtomsDerivatives(qptr[i], j, dqidxj);
            }
        }
    } else {
        //Two domain relative. We're currently in the frame of domain 1.
        //We are rotating group 1 towards group 2. and group 2 towards group 1.
        //The overlap matrix is actually different for arbitrary atom  sets, and will need to be recalculated.
        //Cheat with the concatenation again, and dgroup 1 first, then group 2.
        //To DO: a simplification for homo-dimers like what NAMD would be doing.

        //Do domain 2 first w.r.t. domain 1, as it is the analogue of the single domain case.
        unsigned offset=refpos1.size();
        for (unsigned j=0;j<refpos2.size();j++) {
            //if (refw2[j]==0) continue; //Only apply forces to weighted atoms.

            double const rx = refpos2[j][0];
            double const ry = refpos2[j][1];
            double const rz = refpos2[j][2];

            // dSijdx : derivative of the S matrix, w.r.t. atom x_j
            dSdx[0][0] = Vector(  rx,  ry,  rz);
            dSdx[1][0] = Vector( 0.0, -rz,  ry);
            dSdx[0][1] = dSdx[1][0];
            dSdx[2][0] = Vector(  rz, 0.0, -rx);
            dSdx[0][2] = dSdx[2][0];
            dSdx[3][0] = Vector( -ry,  rx, 0.0);
            dSdx[0][3] = dSdx[3][0];
            dSdx[1][1] = Vector(  rx, -ry, -rz);
            dSdx[2][1] = Vector(  ry,  rx, 0.0);
            dSdx[1][2] = dSdx[2][1];
            dSdx[3][1] = Vector(  rz, 0.0,  rx);
            dSdx[1][3] = dSdx[3][1];
            dSdx[2][2] = Vector( -rx,  ry, -rz);
            dSdx[3][2] = Vector( 0.0,  rz,  ry);
            dSdx[2][3] = dSdx[3][2];
            dSdx[3][3] = Vector( -rx, -ry,  rz);

            // dqi/dxj = Sum_i Sum_j q1i dSijdx q0j /(Norm) * qi...
            Vector dqidxj;
            for (unsigned i=0;i<4;i++) {
                //Calculate and append the derivatives due to each q-component separately
                dqidxj.zero();
                for (unsigned a=0;a<4;a++) {
                    for (unsigned b=0;b<4;b++) {
                        double fact=qmat[1][a]*qmat[0][b]/(eigenvals[0]-eigenvals[1])*qmat[1][i];
                              fact+=qmat[2][a]*qmat[0][b]/(eigenvals[0]-eigenvals[2])*qmat[2][i];
                              fact+=qmat[3][a]*qmat[0][b]/(eigenvals[0]-eigenvals[3])*qmat[3][i];
                        dqidxj+= -1.0*fact*dSdx[a][b];
                    }
                }
                //Rotate back to system frame with q1.
                dqidxj = quaternionRotate(quat1, dqidxj);
                setAtomsDerivatives(qptr[i], j+offset, dqidxj);
            }
        }

        //Now we do domain 1 w.r.t. to domain 2. Q is the inverse, but S is different.
        //First rotate reference 1 to match group 2 frame.
        rotref1 = rotateCoordinates(quat, refpos1, false);
        //Recalculate dSdx for domain 1.
        for (unsigned j=0;j<rotref1.size();j++) {
            //if (refw1[j]==0) continue; //Only apply forces to weighted atoms in the RMSD calculation.

            double const rx = rotref1[j][0];
            double const ry = rotref1[j][1];
            double const rz = rotref1[j][2];

            // dSijdx : derivative of the S matrix, w.r.t. atom x_j
            dSdx[0][0] = Vector(  rx,  ry,  rz);
            dSdx[1][0] = Vector( 0.0,  rz, -ry);
            dSdx[0][1] = dSdx[1][0];
            dSdx[2][0] = Vector( -rz, 0.0,  rx);
            dSdx[0][2] = dSdx[2][0];
            dSdx[3][0] = Vector(  ry, -rx,  0.0);
            dSdx[0][3] = dSdx[3][0];
            dSdx[1][1] = Vector(  rx, -ry, -rz);
            dSdx[2][1] = Vector(  ry,  rx, 0.0);
            dSdx[1][2] = dSdx[2][1];
            dSdx[3][1] = Vector(  rz, 0.0,  rx);
            dSdx[1][3] = dSdx[3][1];
            dSdx[2][2] = Vector( -rx,  ry, -rz);
            dSdx[3][2] = Vector( 0.0,  rz,  ry);
            dSdx[2][3] = dSdx[3][2];
            dSdx[3][3] = Vector( -rx, -ry,  rz);

            // dqi/dxj = Sum_i Sum_j q1i dSijdx q0j /(Norm) * qi...
            Vector dqidxj;
            for (unsigned i=0;i<4;i++) {
                //Calculate and append the derivatives due to each q-component separately
                dqidxj.zero();
                for (unsigned a=0;a<4;a++) {
                    for (unsigned b=0;b<4;b++) {
                        double fact=qmat[1][a]*qmat[0][b]/(eigenvals[0]-eigenvals[1])*qmat[1][i];
                              fact+=qmat[2][a]*qmat[0][b]/(eigenvals[0]-eigenvals[2])*qmat[2][i];
                              fact+=qmat[3][a]*qmat[0][b]/(eigenvals[0]-eigenvals[3])*qmat[3][i];
                        dqidxj+= -1.0*fact*dSdx[a][b];
                    }
                }
                //Rotate back to system frame with q1.
                dqidxj = quaternionRotate(quat1, dqidxj);
                setAtomsDerivatives(qptr[i], j, dqidxj);
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
    #ifdef DEBUG__CHENP
    fprintf (stderr, "= = Debug: QUATERNION::calculate has been called.\n");
    #endif

    // Obtain current position, reference position,
    optimalAlignment1( getPositions() );
    //The values and derivations of the components will have been set
    //within the above function.

  //At the end, transfer atom derivatice from sub-function to the colvar.
  //for(unsigned i=0;i<getNumberOfAtoms();i++) setAtomsDerivatives( i, mypack1.getAtomDerivative(i) );
  //Transfer Virials.
  //Tensor virial; plumed_dbg_assert( !mypack.virialWasSet() );
  //setBoxDerivativesNoPbc();
}

}
}
