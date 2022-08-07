#!/usr/bin/python3

###############################################################################
# Create and save the leadfield matrix                                        #
###############################################################################

import numpy as np
import time

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

mesh_path='../data/tet_mesh.msh'
tensors_path='../data/conductivities.txt'
electrodes_path='../data/electrodes.txt'
dipoles_path='../data/dipoles_ecc_0.99_radial.txt'

def create_leadfield_matrix(mesh_path,tensors_path,electrodes_path,dipoles_path,n,m):
    # create driver
    volume_conductor_cfg = {'grid.filename' : mesh_path, 'tensors.filename' : tensors_path}
    driver_cfg = {'type' : 'fitted', 'solver_type' : 'cg', 'element_type' : 'tetrahedron', 'post_process' : 'true', 'subtract_mean' : 'true'}
    solver_cfg = {'reduction' : '1e-14', 'edge_norm_type' : 'houston', 'penalty' : '20', 'scheme' : 'sipg', 'weights' : 'tensorOnly'}
    driver_cfg['solver'] = solver_cfg
    driver_cfg['volume_conductor'] = volume_conductor_cfg

    print('Creating driver')
    meeg_driver = dp.MEEGDriver3d(driver_cfg)
    print('Driver created')

    # set electrodes
    print('Setting electrodes')
    electrode_cfg = {'type' : 'closest_subentity_center', 'codims' : '3'}
    electrodes = dp.read_field_vectors_3d(electrodes_path)[0:m]
    meeg_driver.setElectrodes(electrodes, electrode_cfg)
    print('Electodes set')

    # compute transfer matrix
    print('Computing transfer matrix')
    transfer_solver_config = {'reduction' : '1e-14'}
    eeg_transfer_config = {'solver' : transfer_solver_config}
    eeg_transfer_matrix, eeg_transfer_computation_information = meeg_driver.computeEEGTransferMatrix(eeg_transfer_config)
    print('Transfer matrix computed')

    # loading dipoles
    print('Reading dipoles')
    dipoles = dp.read_dipoles_3d(dipoles_path)[0:n]
    print(dipoles)
    print('Dipoles read')

    # solve EEG forward problem, which means computing the leadfield
    print('Solving EEG forward problem')
    source_model_cfg = {'type' : 'localized_subtraction', 'restrict' : 'false', 'initialization' : 'single_element', 'intorderadd_eeg_patch' : '0', 'intorderadd_eeg_boundary' : '0', 'intorderadd_eeg_transition' : '0', 'extensions' : 'vertex vertex'}
    driver_cfg['source_model'] = source_model_cfg
    start_time = time.time()
    numerical_solutions, computation_information = meeg_driver.applyEEGTransfer(eeg_transfer_matrix, dipoles, driver_cfg)
    print('EEG forward problem solved')
    print(f"Solving EEG forward problem took {time.time() - start_time} seconds")

    return numerical_solutions


if __name__=="__main__":
    print("Create leadfield matrix")
    n = 100

    m = 20
    L = create_leadfield_matrix(mesh_path,tensors_path,electrodes_path,dipoles_path,n,m)
    np.savez_compressed('../data/leadfield_matrix_10_L1.npz', L)

    m = 40
    L = create_leadfield_matrix(mesh_path,tensors_path,electrodes_path,dipoles_path,n,m)
    np.savez_compressed('../data/leadfield_matrix_10_L2.npz', L)

    m = 70
    L = create_leadfield_matrix(mesh_path,tensors_path,electrodes_path,dipoles_path,n,m)
    np.savez_compressed('../data/leadfield_matrix_10_L3.npz', L)