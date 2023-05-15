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

    # Modified from openmm hdf5.py script
    def _extend_traj_field(self, field_name, field_data):
        """Add one new frames worth of data to the end of an existing
        contiguous (non-sparse)trajectory field.

        Parameters
        ----------

        field_name : str
            Field name
        field_data : numpy.array
            The frames of data to add.
        """

        field = self.h5[field_name]

        # of datase new frames
        n_new_frames = 1

        # check the field to make sure it is not empty
        if all([i == 0 for i in field.shape]):

            feature_dims = field_data.shape
            field.resize((n_new_frames, *feature_dims))

            # set the new data to this
            field[0:, ...] = field_data

        else:
            # append to the dataset on the first dimension, keeping the
            # others the same, these must be feature vectors and therefore
            # must exist
            field.resize((field.shape[0] + n_new_frames, *field_data.shape))
            # add the new data
            field[-n_new_frames:, ...] = field_data

    def describeNextReport(self, simulation):

        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, self._coordinates, self._velosities, self._forces, self._potentialEnerg)

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
