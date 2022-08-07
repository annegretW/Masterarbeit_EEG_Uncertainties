#!/usr/bin/python3

import server
import eeg_model
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

# define different test models
def testmodel_example1():
    return server.TestModel('/home/anne/Masterarbeit/masterarbeit/data/leadfield_matrix_10')

def testmodel_example2():
    return server.TestModel('/home/anne/Masterarbeit/masterarbeit/data/leadfield_matrix_100')

def testmodel_example3():
    # set reference dipol
    s_ref = dp.Dipole3d([127, 127, 197, 0, 0, 92])

    # calc sensor values
    volume_conductor_cfg = {'grid.filename' : mesh_path, 'tensors.filename' : tensors_path}
    driver_cfg = {'type' : 'fitted', 'solver_type' : 'cg', 'element_type' : 'tetrahedron', 'post_process' : 'true', 'subtract_mean' : 'true'}
    solver_cfg = {'reduction' : '1e-14', 'edge_norm_type' : 'houston', 'penalty' : '20', 'scheme' : 'sipg', 'weights' : 'tensorOnly'}
    driver_cfg['solver'] = solver_cfg
    driver_cfg['volume_conductor'] = volume_conductor_cfg
    meeg_driver = dp.MEEGDriver3d(driver_cfg)
    T = transfer_matrix.create_transfer_matrix(mesh_path,tensors_path,electrodes_path)

    source_model_cfg = {'type' : 'localized_subtraction', 'restrict' : 'false', 'initialization' : 'single_element', 'intorderadd_eeg_patch' : '0', 'intorderadd_eeg_boundary' : '0', 'intorderadd_eeg_transition' : '0', 'extensions' : 'vertex vertex'}
    driver_cfg['source_model'] = source_model_cfg
    b_ref, computation_information = meeg_driver.applyEEGTransfer(T, [s_ref], driver_cfg)
    print(b_ref) 
    b_ref = np.random.normal(b_ref, 0.005)
    print(b_ref)

    return eeg_model.EEGModel(b_ref)

# choose a testmodel
testmodel = testmodel_example3()

# send via localhost
umbridge.serve_model(testmodel, 4243) 
