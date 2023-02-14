import MDAnalysis
from MDAnalysis.coordinates import NAMDBIN

u = MDAnalysis.Universe(topology="../inputs/complex.parm7")
u.trajectory =  NAMDBIN.NAMDBINReader('outpats/eq.coor')
all_atoms = u.select_atoms("all")
all_atoms.write('outpats/eq.pdb')