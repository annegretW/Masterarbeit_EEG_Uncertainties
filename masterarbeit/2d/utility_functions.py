import numpy as np
from leadfield import create_leadfield, create_transfer_matrix
import math
import meshio

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp


def save_leadfield_matrix(electrodes_path, conductivities_path, mesh_path, path_leadfield):
    # read mesh
    mesh = meshio.read(mesh_path)

    # set mesh nodes as dipoles
    dipoles = []
    for p in mesh.points:
        dipoles.append(get_radial_dipole(p[0:2],[127,127]))

    # generate leadfield
    leadfield_matrix = create_leadfield(mesh_path , conductivities_path, electrodes_path, dipoles)[1]

    # save leadfield
    np.savez_compressed(path_leadfield, leadfield_matrix)


def save_transfer_matrix(electrodes_path, conductivities_path, mesh_path, path_transfer):
    # read mesh
    mesh = meshio.read(mesh_path)

    # generate transfer matrix
    transfer_matrix = create_transfer_matrix(mesh_path, conductivities_path, electrodes_path)[0]
    
    # save transfer matrix
    np.savez_compressed(path_transfer, transfer_matrix)


def get_radial_dipole_orientation(point, center):
    dim = len(point)
    assert(len(center) == dim)
    if dim == 2:
        r = math.sqrt((point[0]-center[0])**2+(point[1]-center[1])**2)
        if(r==0):
            return [1,0]
        
        rho = math.atan2(point[1]-center[1], point[0]-center[0])
        moment = [np.cos(rho),np.sin(rho)]

        return moment
    else:
        r = math.sqrt((point[0]-center[0])**2+(point[1]-center[1])**2+(point[2]-center[2])**2)
        if(r==0):
            return [1,0,0]
        
        phi = np.arccos((point[2]-center[2])/r)
        rho = math.atan2(point[1]-center[1], point[0]-center[0])
    
        moment = [np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]

        return moment

def get_radial_dipole(point, center):
    dim = len(point)
    assert(len(center) == dim)
    moment = get_radial_dipole_orientation(point, center)
    if dim == 2:
        return dp.Dipole2d(point, moment)
    else:
        return dp.Dipole3d(point, moment)

def get_dipole_orientation(dim, rho, phi=0):
    if dim == 2:
        return [np.cos(rho),np.sin(rho)]
    else:
        return [np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]

def get_dipole(point, center, rho, phi=0):
    dim = len(point)
    assert(len(center) == dim)

    moment = get_dipole_orientation(dim, rho, phi)

    if dim == 2:
        return dp.Dipole2d(point, moment)
    else:
        return dp.Dipole3d(point, moment)

def get_electrodes(mesh_path):
    mesh = meshio.read(mesh_path)
    electrodes = []
    for node in mesh.points:
        if(np.isclose([math.sqrt((node[0]-127)**2+(node[1]-127)**2)],[92],atol=0.01)):
            electrodes.append(node)
    print(len(electrodes))
    np.savez_compressed('data/electrodes', electrodes)

def calc_sensor_values(s_ref, electrodes_path, mesh_path, tensors_path):
    print("s_ref = %s \n" % s_ref)
    #b_ref = analytical_solution(s_ref, mesh_path, tensors_path, electrodes_path)

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

    T, meg_driver = create_transfer_matrix(mesh_path, tensors_path, electrodes_path)
    b_ref = meg_driver.applyEEGTransfer(T,[s_ref],config)[0][0]

    return b_ref


def calc_disturbed_sensor_values(s_ref, electrodes_path, mesh_path, tensors_path, relative_noise):
    b_ref = calc_sensor_values(s_ref, electrodes_path, mesh_path, tensors_path)

    # Disturb sensor values
    sigma = relative_noise*np.amax(np.absolute(b_ref))
    print("sigma = " + str(sigma))
    b_ref = np.random.normal(b_ref, sigma)
    #print("Disturbed measurement values at the electrodes:")
    #print(b_ref) 

    return b_ref, sigma

def find_next_node(nodes, point):
    point = np.array(point)
    x = np.array(nodes-point)
    index = np.linalg.norm(np.absolute(nodes - point), 2, axis=1).argmin()
    #index = np.linalg.norm(np.abs(nodes - point),axis=1).argmin()
    return index