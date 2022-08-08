#!/usr/bin/python3

import umbridge
import numpy as np
import transfer_matrix
from leadfield import create_leadfield
import structured_mesh as msh

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

# mesh_path='../data/tet_mesh.msh'
electrodes_path='../data/electrodes.txt'

class EEGModel(umbridge.Model):
    def __init__(self, b_ref):
        super(EEGModel, self).__init__()

        center = (127,127,127)
        radii = (92,86,80,78)
        conductivities = [0.00043,0.00001,0.00179,0.00033]
        resolutions = [50]

        self.mesh = []
        self.mesh.append(msh.StructuredMesh(center, radii, resolutions[0]))

        self.dipoles = []
        dipoles = []
        # Set dipoles
        for c in self.mesh[0].centers[0]:
            print(c[2]/np.max(self.mesh[0].radii))
            rho = np.arccos((c[2]-self.mesh[0].center[2])/np.max(self.mesh[0].radii))
            phi = np.arctan2(c[1], c[0])
            dipoles.append(dp.Dipole3d(c,[np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]))
        self.dipoles.append(dipoles)

        #self.driver_cfg = config
        #self.meg_driver = meg_driver

        self.transfer_matrix, self.leadfield_matrix = create_leadfield(self.mesh[0], conductivities, electrodes_path, self.dipoles[0])

        # reference values of sources
        self.b_ref = b_ref


    def get_input_sizes(self):
        return [3]

    def get_output_sizes(self):
        return [1]

    def posterior(self, theta, level):
        sigma = 0.005
        b = self.leadfield_matrix[self.mesh[0].find_next_node(theta)]

        posterior = -(1/(2*sigma**2))*(np.linalg.norm(np.array(self.b_ref)-np.array(b), 2))**2

        return posterior

    def __call__(self, parameters, config={'level': 1}):
        return [[self.posterior(parameters[0],config["level"])]]

    def supports_evaluate(self):
        return True

