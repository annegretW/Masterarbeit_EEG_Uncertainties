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
        dipoles.append(get_dipole(p[0:2],[127,127]))

    # generate leadfield
    leadfield_matrix = create_leadfield(mesh_path , conductivities_path, electrodes_path, dipoles)[1]

    # save leadfield
    np.savez_compressed(path_leadfield, leadfield_matrix)


def get_dipole(point, center):
    dim = len(point)
    assert(len(center) == dim)
    if dim == 2:
        r = math.sqrt((point[0]-center[0])**2+(point[1]-center[1])**2)
        if(r==0):
            return dp.Dipole2d(center, [1,0])
        
        rho = math.atan2(point[1]-center[1], point[0]-center[0])
    
        moment = [np.cos(rho),np.sin(rho)]

        return dp.Dipole2d(point, moment)
    else:
        r = math.sqrt((point[0]-center[0])**2+(point[1]-center[1])**2+(point[2]-center[2])**2)
        if(r==0):
            return dp.Dipole3d(center, [1,0,0])
        
        phi = np.arccos((point[2]-center[2])/r)
        rho = math.atan2(point[1]-center[1], point[0]-center[0])
    
        moment = [np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]

        return dp.Dipole3d(point, moment)

def get_electrodes(mesh):
    electrodes = []
    for node in mesh.points:
        if(np.isclose([math.sqrt((node[0]-127)**2+(node[1]-127)**2)],[92],atol=0.01)):
            electrodes.append(node)
    print(len(electrodes))
    np.savez_compressed('data/electrodes', electrodes)

def calc_disturbed_sensor_values(s_ref, electrodes_path, relative_noise):
    print("#################################################################")
    print("Simulate disturbed sensor values for a given dipole.")
    print("################################################################# \n")

    print("s_ref = %s \n" % s_ref)
    mesh_path = "data/mesh_3.msh"
    tensors_path = "data/conductivities.txt"
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