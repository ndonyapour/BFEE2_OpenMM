import os
import numpy as np
import h5py
import pickle as pkl
import simtk.unit as unit
from collections import defaultdict
import openmm.openmm as omm

MAX_GRIDS = 20000

    # def getHillHeight(self, simulation):
    #     """Get the current height of the Gaussian hill in kJ/mol"""
    #     energy = simulation.context.getState(getEnergy=True, groups={31}).getPotentialEnergy()
    #     currentHillHeight = self.height*np.exp(-energy/(unit.MOLAR_GAS_CONSTANT_R*self._deltaT))
    #   return ((self.temperature+self._deltaT)/self._deltaT)*currentHillHeight.value_in_unit(unit.kilojoules_per_mole)
class HILLSReporter(object):

    def __init__(self, meta, traj_file_path, reportInterval=100, CV=True, BIAS=True):
        self.traj_file_path = traj_file_path
        self._h5 = None
        self._reportInterval = reportInterval
        self._is_intialized = False
        self._CV = CV
        self._BIAS = BIAS
        self.meta = meta 

    def _initialize(self, simulation):

        self.h5 = h5py.File(self.traj_file_path, 'w', libver='latest')
        self.h5.swmr_mode = True

        if self._CV:
            self.h5.create_dataset('cv', (0, 0), maxshape=(None, MAX_GRIDS))
        if self._BIAS:
            self.h5.create_dataset('bias', (0, 0, 0), maxshape=(None, MAX_GRIDS, MAX_GRIDS))

    def describeNextReport(self, simulation):

        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps)

    def report(self, simulation, state):

        if not self._is_intialized:
            self._initialize(simulation)
            self._is_intialized = True

        if self._CV:
            cv = self.meta.getCollectiveVariables(simulation)
            self._extend_traj_field('cv', np.array(cv))
        if self._BIAS:
            bias = self.meta._selfBias
            self._extend_traj_field('bias', np.array(bias))
            
        self.h5.flush()

    def close(self):
        "Close the underlying trajectory file"
        self.h5.close()


    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, True, True, False)
