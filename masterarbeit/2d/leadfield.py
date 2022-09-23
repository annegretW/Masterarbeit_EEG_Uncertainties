import numpy as np

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp


def create_leadfield(mesh_path, tensors_path, electrodes_path, dipoles):
    # Create MEG driver
    config = {
        'type' : 'fitted',
        'solver_type' : 'cg',
        'element_type' : 'tetrahedron',
        'volume_conductor': {
            'grid.filename' : mesh_path, 
            'tensors.filename' : tensors_path
         },
         'post_process' : 'true', 
         'subtract_mean' : 'true'
    }
    meg_driver = dp.MEEGDriver2d(config)

    # Set electrode positions
    # electrodes = np.genfromtxt(electrodes_path,delimiter=None) 
    electrodes = np.load(electrodes_path)["arr_0"]
    electrodes = electrodes[:,0:2] # ensure 2d
    electrodes = [dp.FieldVector2D(t) for t in electrodes.tolist()]
    #electrode_config = {
    #    'type' : 'closest_subentity_center',
    #    'codims' : [3]
    #}
    electrode_config = {'type': 'normal'}
    meg_driver.setElectrodes(electrodes, electrode_config)

    # Compute transfer matrix
    transfer_solver_config = {'reduction' : '1e-14'}
    eeg_transfer_config = {'solver' : transfer_solver_config}
    transfer_matrix, eeg_transfer_computation_information = meg_driver.computeEEGTransferMatrix(eeg_transfer_config)

    # Compute leadfield matrix
    source_model_config_v = {
        'type' : 'venant',
        'numberOfMoments' : 2,
        'referenceLength' : 20,
        'weightingExponent' : 1,
        'relaxationFactor' : 1e-6,
        'mixedMoments' : True,
        'restrict' : True,
        'initialization' : 'closest_vertex'
    }
    
    source_model_config_s = {
        'type' : 'subtraction',
        'intorderadd' : 2,
        'intorderadd_lb' : 2}

    driver_config = {
        'solver.reduction' : 1e-10,
        'source_model' : source_model_config_v,
        'post_process' : True,
        'subtract_mean' : True
    }

    leadfield_matrix, computation_information = meg_driver.applyEEGTransfer(transfer_matrix, dipoles, driver_config)

    return transfer_matrix, leadfield_matrix

def create_transfer_matrix(mesh_path, tensors_path, electrodes_path):
    config = {
        'type' : 'fitted',
        'solver_type' : 'cg',
        'element_type' : 'tetrahedron',
        'volume_conductor': {
            'grid.filename' : mesh_path, 
            'tensors.filename' : tensors_path
         },
         'post_process' : 'true', 
         'subtract_mean' : 'true'
    }
    meg_driver = dp.MEEGDriver2d(config)

    # Set electrode positions
    # electrodes = np.genfromtxt(electrodes_path,delimiter=None) 
    electrodes = np.load(electrodes_path)["arr_0"]
    electrodes = electrodes[:,0:2] # ensure 2d
    electrodes = [dp.FieldVector2D(t) for t in electrodes.tolist()]
    #electrode_config = {
    #    'type' : 'closest_subentity_center',
    #    'codims' : [3]
    #}
    electrode_config = {'type': 'normal'}
    meg_driver.setElectrodes(electrodes, electrode_config)

    # Compute transfer matrix
    transfer_solver_config = {'reduction' : '1e-14'}
    eeg_transfer_config = {'solver' : transfer_solver_config}
    transfer_matrix, eeg_transfer_computation_information = meg_driver.computeEEGTransferMatrix(eeg_transfer_config)

    return transfer_matrix, meg_driver