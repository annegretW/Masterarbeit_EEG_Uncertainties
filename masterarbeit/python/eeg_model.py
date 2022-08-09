#!/usr/bin/python3

import umbridge
import numpy as np
import transfer_matrix
from leadfield import create_leadfield, create_transfer_matrix
import structured_mesh as msh

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

electrodes_path='../data/electrodes.txt'

class EEGModel(umbridge.Model):
    def __init__(self, b_ref):
        super(EEGModel, self).__init__()

        # parameters
        center = (127,127,127)
        radii = (92,86,80,78)
        conductivities = [0.00043,0.00001,0.00179,0.00033]
        cells_per_dim = [16, 32, 64]
        sigma = 0.005

        # initialize lists
        self.mesh = []
        dipoles_list = []
        self.transfer_matrix = []
        self.leadfield_matrix = []

        for i in range(len(cells_per_dim)):
            # create mesh
            self.mesh.append(msh.StructuredMesh(center, radii, cells_per_dim[i]))

            # set dipoles
            dipoles = []
            for c in self.mesh[i].centers[0]:
                rho = np.arccos((c[2]-self.mesh[i].center[2])/np.max(self.mesh[i].radii))
                phi = np.arctan2(c[1]-self.mesh[i].center[1], c[0]-self.mesh[i].center[0])
                dipoles.append(dp.Dipole3d(c,[np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]))
            dipoles_list.append(dipoles)

            # create transfer matrix and leadfield matrix
            T, L = create_leadfield(self.mesh[i], conductivities, electrodes_path, dipoles_list[i])
            self.transfer_matrix.append(T)
            self.leadfield_matrix.append(L)

        # reference values of sources
        self.b_ref = b_ref

        # posterior variance
        self.sigma = sigma


    def get_input_sizes(self):
        return [3]

    def get_output_sizes(self):
        return [1]

    def posterior(self, theta, level):
        i = level-1
        b = self.leadfield_matrix[i][self.mesh[i].find_next_node(theta)]

        posterior = -(1/(2*self.sigma**2))*(np.linalg.norm(np.array(self.b_ref)-np.array(b), 2))**2

        return posterior

    def __call__(self, parameters, config={'level': 1}):
        return [[self.posterior(parameters[0],config["level"])]]

    def supports_evaluate(self):
        return True

class EEGModelNew(umbridge.Model):
    def __init__(self, b_ref):
        super(EEGModelNew, self).__init__()

        # parameters
        center = (127,127,127)
        radii = (92,86,80,78)
        conductivities = [0.00043,0.00001,0.00179,0.00033]
        cells_per_dim = [16, 32, 64]
        sigma = 0.005

        # initialize lists
        self.mesh = []
        dipoles_list = []
        self.transfer_matrix = []
        self.meg_drivers = []

        for i in range(len(cells_per_dim)):
            # create mesh
            self.mesh.append(msh.StructuredMesh(center, radii, cells_per_dim[i]))

            # set dipoles
            dipoles = []
            for c in self.mesh[i].centers[0]:
                rho = np.arccos((c[2]-self.mesh[i].center[2])/np.max(self.mesh[i].radii))
                phi = np.arctan2(c[1]-self.mesh[i].center[1], c[0]-self.mesh[i].center[0])
                dipoles.append(dp.Dipole3d(c,[np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]))
            dipoles_list.append(dipoles)

            # create transfer matrix and leadfield matrix
            T, meg_driver = create_transfer_matrix(self.mesh[i], conductivities, electrodes_path)
            self.transfer_matrix.append(T)
            self.meg_drivers.append(meg_driver)

        # reference values of sources
        self.b_ref = b_ref

        # posterior variance
        self.sigma = sigma

        self.center = center
        self.radii = radii


    def get_input_sizes(self):
        return [3]

    def get_output_sizes(self):
        return [1]

    def posterior(self, theta, level):
        i = level-1
        next_dipole_pos = self.mesh[i].find_next_center(theta) 

        dist = np.sqrt(sum([(x-c)**2 for x,c in zip(next_dipole_pos, self.center)]))

        if dist > np.max(self.radii):
            return 0

        rho = np.arccos((next_dipole_pos[2]-self.mesh[i].center[2])/np.max(self.mesh[i].radii))
        phi = np.arctan2(next_dipole_pos[1]-self.mesh[i].center[1], next_dipole_pos[0]-self.mesh[i].center[0])
        next_dipole = dp.Dipole3d(next_dipole_pos,[np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)])

        source_model_config = {
        'type' : 'venant',
        'numberOfMoments' : 3,
        'referenceLength' : 20,
        'weightingExponent' : 1,
        'relaxationFactor' : 1e-6,
        'mixedMoments' : True,
        'restrict' : True,
        'initialization' : 'closest_vertex'
        }

        config = {
            'solver.reduction' : 1e-10,
            'source_model' : source_model_config,
            'post_process' : True,
            'subtract_mean' : True
        }

        b = self.meg_drivers[i].applyEEGTransfer(self.transfer_matrix[i],[next_dipole],config)[0]

        posterior = -(1/(2*self.sigma**2))*(np.linalg.norm(np.array(self.b_ref)-np.array(b), 2))**2

        return posterior

    def __call__(self, parameters, config={'level': 1}):
        return [[self.posterior(parameters[0],config["level"])]]

    def supports_evaluate(self):
        return True

