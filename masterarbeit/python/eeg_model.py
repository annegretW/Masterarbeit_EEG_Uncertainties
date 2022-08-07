#!/usr/bin/python3

import umbridge
import numpy as np
import transfer_matrix

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

mesh_path='../data/tet_mesh.msh'
tensors_path='../data/conductivities.txt'
electrodes_path='../data/electrodes.txt'

class EEGModel(umbridge.Model):
    def __init__(self, b_ref):
        super(EEGModel, self).__init__()

        volume_conductor_cfg = {'grid.filename' : mesh_path, 'tensors.filename' : tensors_path}
        self.driver_cfg = {'type' : 'fitted', 'solver_type' : 'cg', 'element_type' : 'tetrahedron', 'post_process' : 'true', 'subtract_mean' : 'true'}
        solver_cfg = {'reduction' : '1e-14', 'edge_norm_type' : 'houston', 'penalty' : '20', 'scheme' : 'sipg', 'weights' : 'tensorOnly'}
        self.driver_cfg['solver'] = solver_cfg
        self.driver_cfg['volume_conductor'] = volume_conductor_cfg
        self.meeg_driver = dp.MEEGDriver3d(self.driver_cfg)

        source_model_cfg = {'type' : 'partial_integration', 'restrict' : 'false', 'initialization' : 'single_element', 'intorderadd_eeg_patch' : '0', 'intorderadd_eeg_boundary' : '0', 'intorderadd_eeg_transition' : '0', 'extensions' : 'vertex vertex'}
        self.driver_cfg['source_model'] = source_model_cfg

        # transfer matrix
        self.T = transfer_matrix.create_transfer_matrix(mesh_path,tensors_path,electrodes_path)

        # reference values of sources
        self.b_ref = b_ref

    def get_input_sizes(self):
        return [6]

    def get_output_sizes(self):
        return [1]

    def posterior(self, theta, level):
        sigma = 0.005
        d = dp.Dipole3d(theta)
        b, computation_information = self.meeg_driver.applyEEGTransfer(self.T, [d], self.driver_cfg)
        posterior = -(1/(2*sigma**2))*(np.linalg.norm(self.b_ref-b, 2))**2

        return posterior

    def __call__(self, parameters, config={'electrodes': 1020}):
        return [[self.posterior(parameters[0],config["electrodes"])]]

    def supports_evaluate(self):
        return True

