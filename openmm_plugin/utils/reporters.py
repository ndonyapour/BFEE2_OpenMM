import os
import numpy as np
import simtk.unit as unit
import h5py


MAX_GRIDS = 20000

class HILLSReporter(object):

    def __init__(self, meta, file_path, sigma, reportInterval=1000, cvname="rmsd", mod="w"):
        self.file_path = os.path.join(file_path, "HILLS")
        self._file = None
        self._reportInterval = reportInterval
        self.sigma = sigma
        self._is_intialized = False
        self.mod = mod
        self.cvname = cvname
        self.meta = meta
        self.forceGroup = meta._force.getForceGroup()
        self.bias = meta.biasFactor

    def _initialize(self, simulation):

        self._file = open(self.file_path, self.mod)
        self._file.write(f'#! FIELDS time {self.cvname} sigma_{self.cvname} height biasf\n')
        self._file.write('#! SET multivariate false\n')
        self._file.write('#! SET kerneltype gaussian\n')
        self._file.flush()
        
    def describeNextReport(self, simulation):

        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return  (steps, False, True, True, False)

    def report(self, simulation, state):
        """Get the current height of the Gaussian hill in kJ/mol"""

        if not self._is_intialized:
            self._initialize(simulation)
            self._is_intialized = True
            
        state = simulation.context.getState(getEnergy=True, groups={self.forceGroup})
        energy = state.getPotentialEnergy()
        currentHillHeight = self.meta.height*np.exp(-energy/(unit.MOLAR_GAS_CONSTANT_R*self.meta._deltaT))
        hillhight = ((self.meta.temperature+self.meta._deltaT)/self.meta._deltaT)*currentHillHeight.value_in_unit(unit.kilojoules_per_mole)

        time = state.getTime().value_in_unit(unit.picosecond)
        time = round(time, 4)
        cv = self.meta.getCollectiveVariables(simulation)
        self._file.write(f'{time:15} {cv[0]:20.16f}          {self.sigma} {hillhight:20.16f}          {self.bias}\n')            
        self._file.flush()
        
    def close(self):
        "Close the underlying trajectory file"
        self._file.close()


    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, True, True, False)


class COLVARReporter(object):

    def __init__(self, meta, file_path, forces, reportInterval=1000):
        self.file_path = file_path
        self._h5 = None
        self._reportInterval = reportInterval
        self._is_intialized = False
        self.meta = meta 
        self.forces = forces

    def _initialize(self, simulation):

        self.h5 = h5py.File(os.path.join(self.file_path, "COLVAR.h5"), 'w', libver='latest')
        self.h5.swmr_mode = True
        self.h5.create_dataset('CV', (0, 0), maxshape=(None, MAX_GRIDS))

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
        return (steps, False, True, True, False)

    def report(self, simulation, state):

        if not self._is_intialized:
            self._initialize(simulation)
            self._is_intialized = True

        current_cvs = list(self.meta.getCollectiveVariables(simulation))
        for f in self.forces:
            current_cvs.extend(f.getCollectiveVariableValues(simulation.context))

        self._extend_traj_field('CV', np.array(current_cvs))
    
        self.h5.flush()

    def close(self):
        "Close the underlying trajectory file"
        self.h5.close()


    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, True, True, False)