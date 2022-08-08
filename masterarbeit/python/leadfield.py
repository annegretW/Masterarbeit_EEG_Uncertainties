import numpy as np

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp


def create_leadfield(mesh, conductivities, electrodes_path, dipoles):
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
    transfer_matrix, eeg_transfer_computation_information = meg_driver.computeEEGTransferMatrix(eeg_transfer_config)


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

    leadfield_matrix, computation_information = meg_driver.applyEEGTransfer(transfer_matrix, dipoles, driver_config)

    return transfer_matrix, leadfield_matrix