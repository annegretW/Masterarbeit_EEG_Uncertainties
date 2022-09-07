import numpy as np
import structured_mesh as msh
from leadfield import create_leadfield, create_transfer_matrix
import transfer_matrix

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

mesh_path='../data/tet_mesh.msh'
tensors_path='../data/conductivities.txt'
electrodes_path='../data/electrodes_1005.txt'

def save_leadfield_matrix(path_mesh, path_leadfield, cells_per_dim):
    conductivities = [0.00043,0.00001,0.00179,0.00033]
    center = (127,127,127)
    radii = (92,86,80,78)

    # create mesh
    mesh = msh.StructuredMesh(center, radii, cells_per_dim)
    np.savez_compressed(path_mesh, mesh, allow_pickle=True)

    # set dipoles
    dipoles = []
    for c in mesh.centers[0]:
        rho = np.arccos((c[2]-mesh.center[2])/np.max(mesh.radii))
        phi = np.arctan2(c[1]-mesh.center[1], c[0]-mesh.center[0])
        dipoles.append(dp.Dipole3d(c,[np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]))

    leadfield_matrix = create_leadfield(mesh, conductivities, electrodes_path, dipoles)[0]
    np.savez_compressed(path_leadfield, leadfield_matrix)


def save_transfer_matrix(path_mesh, electrodes_path, path_transfer_matrix, cells_per_dim):
    conductivities = [0.00043,0.00001,0.00179,0.00033]
    center = (127,127,127)
    radii = (92,86,80,78)

    # create mesh
    mesh = msh.StructuredMesh(center, radii, cells_per_dim)
    #np.savez_compressed(path_mesh, mesh, allow_pickle=True)

    # set dipoles
    dipoles = []
    for c in mesh.centers[0]:
        rho = np.arccos((c[2]-mesh.center[2])/np.max(mesh.radii))
        phi = np.arctan2(c[1]-mesh.center[1], c[0]-mesh.center[0])
        dipoles.append(dp.Dipole3d(c,[np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]))

    transfer_matrix = create_transfer_matrix(mesh, conductivities, electrodes_path)[0]

    np.savez_compressed(path_transfer_matrix, transfer_matrix)
    

def get_dipole(point):
    rho = np.arccos((point[2]-127)/92)
    phi = np.arctan2(point[1]-127, point[0]-127)
    moment = [np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]
    
    return dp.Dipole3d(point, moment)


def calc_disturbed_sensor_values(s_ref, electrodes_path):
    print("#################################################################")
    print("Simulate disturbed sensor values for a given dipole.")
    print("################################################################# \n")

    print("s_ref = %s \n" % s_ref)

    # Calc correct sensor values
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
    print("Calculated measurement values at the electrodes:")
    print(b_ref) 
    print("\n")

    # Disturb sensor values
    b_ref = np.random.normal(b_ref, 0.005)
    print("Disturbed measurement values at the electrodes:")
    print(b_ref) 

    return b_ref