#!/usr/bin/python3

###############################################################################
# Create and save the leadfield matrix                                        #
###############################################################################

import numpy as np

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

mesh_path='../data/tet_mesh.msh'
tensors_path='../data/conductivities.txt'
electrodes_path='../data/electrodes.txt'

def create_transfer_matrix(mesh_path,tensors_path,electrodes_path):
    # create driver
    volume_conductor_cfg = {
        'grid.filename' : mesh_path, 
        'tensors.filename' : tensors_path
        }
    driver_cfg = {
        'type' : 'fitted', 
        'solver_type' : 'cg', 
        'element_type' : 'tetrahedron', 
        'post_process' : True, 
        'subtract_mean' : True
        }
    solver_cfg = {
        'reduction' : '1e-14', 
        'edge_norm_type' : 'houston', 
        'penalty' : '20', 
        'scheme' : 'sipg', 
        'weights' : 'tensorOnly'}
        
    driver_cfg['solver'] = solver_cfg
    driver_cfg['volume_conductor'] = volume_conductor_cfg

    print('Creating driver')
    meeg_driver = dp.MEEGDriver3d(driver_cfg)
    print('Driver created')

    # set electrodes
    print('Setting electrodes')
    electrode_cfg = {'type' : 'closest_subentity_center', 'codims' : '3'}
    electrodes = dp.read_field_vectors_3d(electrodes_path)
    meeg_driver.setElectrodes(electrodes, electrode_cfg)
    print('Electodes set')

    # compute transfer matrix
    print('Computing transfer matrix')
    transfer_solver_config = {'reduction' : '1e-14'}
    eeg_transfer_config = {'solver' : transfer_solver_config}
    eeg_transfer_matrix, eeg_transfer_computation_information = meeg_driver.computeEEGTransferMatrix(eeg_transfer_config)
    print('Transfer matrix computed')

    return eeg_transfer_matrix