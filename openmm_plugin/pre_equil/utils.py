from sys import stdout
import time
import openmm.app as omma
import openmm as omm
import simtk.unit as unit
from openmmtools.integrators import GradientDescentMinimizationIntegrator



def GD_minimize(prmtop, platform, properties, coords, num_steps=10000):

    # prepare system
    system = prmtop.createSystem(nonbondedMethod=omma.PME,
                                 nonbondedCutoff=1*unit.nanometer,
                                 constraints=omma.HBonds,
                                 ewaldErrorTolerance=0.0005)
    
    integrator = GradientDescentMinimizationIntegrator(initial_step_size=0.0001)

    simulation = omma.Simulation(prmtop.topology, system,
                                integrator, platform, properties)

    simulation.context.setPositions(coords)
    print('Start Gradient Descent Minimizer') 
    print(f'Energy before minimization = {simulation.context.getState(getEnergy=True).getPotentialEnergy()}')
    simulation.step(num_steps)
    print('Finish Gradient Descent Minimizer')

    return simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    
    


def LBFGS_minimize(prmtop, platform, properties, coords, num_steps=10000):

   # prepare system
    system = prmtop.createSystem(nonbondedMethod=omma.PME,
                                 nonbondedCutoff=1*unit.nanometer,
                                 constraints=omma.HBonds,
                                 ewaldErrorTolerance=0.0005)

    integrator = omm.LangevinIntegrator(300*unit.kelvin, 
                                        1.0/unit.picoseconds,
                                        0.002*unit.picoseconds)
   
    simulation = omma.Simulation(prmtop.topology, system,
                                integrator, platform, properties)
    simulation.context.setPositions(coords)
    
    print('Start L-BFGS Minimizer')
    simulation.minimizeEnergy(maxIterations=num_steps)
    print('Finish L-BFGS Minimizer')
    print(f'Energy after minimization = {simulation.context.getState(getEnergy=True).getPotentialEnergy()}')

    return simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()


def minimize(prmtop, platform, properties, coords, num_steps=10000, save_path='outputs/complex_min.rst'):
    start_time = time.time()
    gd_min_pos = GD_minimize(prmtop, platform, properties, coords, num_steps)
    min_pos = LBFGS_minimize(prmtop, platform, properties, gd_min_pos, num_steps)
    end_time = time.time()
    print(f"Minimization Run time = {round(end_time - start_time, 3)} s")

    # save minimized positions
    prmtop.coordinates = min_pos
    prmtop.save(save_path, format='rst7', overwrite=True)
    
    return min_pos

def equil_NVT(prmtop, platform, properties, coords, num_steps=10000, save_path='outputs/complex_nvt.rst', 
              log_reporter= 'NVT_log', log_reporter_steps=500):

    # prepare system
    system = prmtop.createSystem(nonbondedMethod=omma.PME,
                                 nonbondedCutoff=1*unit.nanometer,
                                 constraints=omma.HBonds,
                                 ewaldErrorTolerance=0.0005)

    integrator = omm.LangevinIntegrator(300*unit.kelvin, 
                                        1.0/unit.picoseconds,
                                        0.002*unit.picoseconds)
   
    simulation = omma.Simulation(prmtop.topology, system,
                                integrator, platform, properties)
    simulation.context.setPositions(coords)
    simulation.reporters.append(
    omma.StateDataReporter(
        log_reporter,
        log_reporter_steps,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        volume=True,
        temperature=True,
        totalSteps=True,
        separator=" ")
    )
    

    start_time = time.time()
    
    print('Start NVT equilibration')
    simulation.step(num_steps)
    print('Finish NVT equilibration')
    print(f'Energy after NVT  equilibration = {simulation.context.getState(getEnergy=True).getPotentialEnergy()}')

    end_time = time.time()
    print(f"NVT Run time = {round(end_time - start_time, 3)} s")
    
    equil_coords = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    prmtop.coordinates = equil_coords
    prmtop.save(save_path, format='rst7', overwrite=True)

    return equil_coords


def equil_NPT(prmtop, platform, properties, coords, num_steps=10000, save_path='outputs/complex_npt.rst', 
              log_reporter= 'NPT_log', log_reporter_steps=10000):
        # prepare system
    system = prmtop.createSystem(nonbondedMethod=omma.PME,
                                 nonbondedCutoff=1*unit.nanometer,
                                 constraints=omma.HBonds,
                                 ewaldErrorTolerance=0.0005)

    barostat = omm.MonteCarloBarostat(1.0*unit.atmosphere, 300*unit.kelvin,  50)
    system.addForce(barostat)
    
    integrator = omm.LangevinIntegrator(300*unit.kelvin, 
                                        1.0/unit.picoseconds,
                                        0.002*unit.picoseconds)
   
    simulation = omma.Simulation(prmtop.topology, system,
                                integrator, platform, properties)
    simulation.context.setPositions(coords)

    simulation.reporters.append(
    omma.StateDataReporter(
        log_reporter,
        log_reporter_steps,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        volume=True,
        temperature=True,
        totalSteps=True,
        separator=" ")
    )


    start_time = time.time()
    
    print('Start NPT equilibration')
    simulation.step(num_steps)
    print('Finish NPT equilibration')
    print(f'Energy after NPT  equilibration = {simulation.context.getState(getEnergy=True).getPotentialEnergy()}')

    end_time = time.time()
    print(f"NPT Run time = {round(end_time - start_time, 3)} s")
    
    equil_coords = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    prmtop.coordinates = equil_coords
    prmtop.save(save_path, format='rst7', overwrite=True)
    
    return equil_coords
