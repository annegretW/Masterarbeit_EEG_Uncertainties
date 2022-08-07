#!/usr/bin/python3

import umbridge
import numpy as np
import transfer_matrix

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

# mesh_path='../data/tet_mesh.msh'
electrodes_path='../data/electrodes.txt'

class EEGModel(umbridge.Model):
    def __init__(self, b_ref, mesh, conductivities):
        super(EEGModel, self).__init__()

        self.mesh = mesh

        # Create MEG driver
        config = {
            'type' : 'fitted',
            'solver_type' : 'cg',
            'element_type' : 'hexahedron',
            'volume_conductor' : {
                'grid' : {
                    'elements' : mesh.elements.tolist(),
                    'nodes' : mesh.nodes.tolist()
                },
                'tensors' : {
                    'labels' : mesh.labels.tolist(),
                    'conductivities' : conductivities
                }
            },
            'post_process' : 'true', 
            'subtract_mean' : 'true'
        }
        meg_driver = dp.MEEGDriver3d(config)

        # Set electrode positions
        electrodes = np.genfromtxt(electrodes_path,delimiter=None) 
        electrodes = [dp.FieldVector3D(t) for t in electrodes.tolist()]
        electrode_config = {
            'type' : 'closest_subentity_center',
            'codims' : [3]
        }
        meg_driver.setElectrodes(electrodes, electrode_config)

        # Compute EEG leadfield
        source_model_cfg = {
            'type' : 'partial_integration',
            'numberOfMoments' : 3,
            'referenceLength' : 20,
            'weightingExponent' : 1,
            'relaxationFactor' : 1e-6,
            'mixedMoments' : True,
            'restrict' : True,
            'initialization' : 'closest_vertex'
        }

        # Compute transfer matrix
        transfer_solver_config = {'reduction' : '1e-14'}
        eeg_transfer_config = {'solver' : transfer_solver_config}
        self.transfer_matrix, eeg_transfer_computation_information = meg_driver.computeEEGTransferMatrix(eeg_transfer_config)

        self.dipoles = []
        # Set dipoles
        for c in mesh.centers[0]:
            print(c[2]/np.max(mesh.radii))
            rho = np.arccos(c[2]/np.max(mesh.radii))
            phi = np.arctan2(c[1], c[0])
            self.dipoles.append(dp.Dipole3d(c,[np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]))

        # Compute leadfield matrix
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

        driver_config = {
            'solver.reduction' : 1e-10,
            'source_model' : source_model_config,
            'post_process' : True,
            'subtract_mean' : True
        }

        print("Dipoles: " + str(len(self.dipoles)))
        print("Electrodes: " + str(len(electrodes)))

        self.leadfield_matrix, computation_information = meg_driver.applyEEGTransfer(self.transfer_matrix, self.dipoles, driver_config)
        print(len(self.leadfield_matrix[0]))

        self.driver_cfg = config
        self.meg_driver = meg_driver

        # reference values of sources
        self.b_ref = b_ref


    def get_input_sizes(self):
        return [3]

    def get_output_sizes(self):
        return [1]

    def posterior(self, theta, level):
        sigma = 0.005
        b = self.leadfield_matrix[self.mesh.find_next_node(theta)]

        print(self.b_ref)
        print(b)
        
        posterior = -(1/(2*sigma**2))*(np.linalg.norm(self.b_ref-b, 2))**2

        return posterior

    def __call__(self, parameters, config={'level': 1}):
        return [[self.posterior(parameters[0],config["level"])]]

    def supports_evaluate(self):
        return True

