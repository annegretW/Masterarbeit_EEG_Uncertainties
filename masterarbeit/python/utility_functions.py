import numpy as np
import structured_mesh as msh
from leadfield import create_leadfield, create_transfer_matrix
from eeg_transfer_approach_mesh_via_file import analytical_solution
import transfer_matrix
import math
duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp


def save_leadfield_matrix(electrodes_path, path_leadfield, cells_per_dim):
    conductivities = [0.00043,0.00001,0.00179,0.00033]
    center = (127,127,127)
    radii = (92,86,80,78)

    # create mesh
    mesh = msh.StructuredMesh(center, radii, cells_per_dim)
    #np.savez_compressed(path_mesh, mesh, allow_pickle=True)

    # set dipoles
    dipoles = []
    for c in mesh.centers[0]:
       dipoles.append(get_dipole(c))

    # dipoles = [get_dipole([110,120,130]),get_dipole([130,120,110])]
    
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
        dipoles.append(get_dipole(c))

    transfer_matrix = create_transfer_matrix(mesh, conductivities, electrodes_path)[0]

    np.savez_compressed(path_transfer_matrix, transfer_matrix)
    

def get_dipole(point):
    r = math.sqrt((point[0]-127)**2+(point[1]-127)**2+(point[2]-127)**2)

    if(r==0):
        return dp.Dipole3d([127,127,127], [1,0,0])
        
    phi = np.arccos((point[2]-127)/r)
    rho = math.atan2(point[1]-127, point[0]-127)
    
    moment = [np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]

    return dp.Dipole3d(point, moment)


def calc_disturbed_sensor_values(s_ref, electrodes_path):
    print("#################################################################")
    print("Simulate disturbed sensor values for a given dipole.")
    print("################################################################# \n")

    print("s_ref = %s \n" % s_ref)
    mesh_path = "../data/tet_mesh.msh"
    tensors_path = "../data/electrodes_1005.txt"
    b_ref = analytical_solution(s_ref, mesh_path, tensors_path, electrodes_path)

    # Disturb sensor values
    sigma = 0.005*np.amax(np.absolute(b_ref))
    print("sigma = " + str(sigma))
    #b_ref = np.random.normal(b_ref, sigma)
    #print("Disturbed measurement values at the electrodes:")
    #print(b_ref) 

    return b_ref, sigma

def saveStructuredHexMeshForSphere(N, center, radii, path):
    nr_nodes = N+1
    lower_left = center - np.max(radii)
    upper_right = center + np.max(radii)
    # create nodes array for bounding box
    nodes = np.meshgrid(*[np.linspace(ll,ur,num=N+1) for ll,ur in zip(lower_left, upper_right)], indexing='ij')
    nodes_array = np.reshape(np.transpose(np.array(nodes, order='F')),[nr_nodes**3, 3], order = 'C')
    # map elements to nodes in bounding box
    elements = np.empty(shape=(N**3,8))
    idx_nodes = np.arange(nodes_array.shape[0])
    idx_back = idx_nodes[-nr_nodes**2:idx_nodes.size]
    idx_right = idx_nodes[idx_nodes%nr_nodes==nr_nodes-1]
    idx_top =  idx_nodes[idx_nodes%nr_nodes**2  >= (nr_nodes)*(nr_nodes-1) ]
    elements[:,0] = np.delete(idx_nodes,np.append(idx_back,np.append(idx_right, idx_top)))
    elements[:,1]     = elements[:,0] + 1
    elements[:,[2,3]] = elements[:,[0,1]] + nr_nodes
    elements[:,4:]    = elements[:,:4] + nr_nodes**2
    nr_elements = len(elements)

    # create tissue labels for elements in bounding box
    centers = np.meshgrid(*[ll+(ur-ll)*(np.linspace(1/N,1,N)-.5/N) for ll,ur in zip(lower_left, upper_right)], indexing = 'ij')
    dist = np.sqrt(sum([(x-c)**2 for x,c in zip(centers, center)]))
    labels = np.zeros([N]*len(lower_left), dtype=int)
    for i, r in enumerate(sorted(radii, reverse=True)):
        labels[dist<r] = i+1

    centers_array = np.reshape(np.transpose(np.array(centers, order='F')),[nr_elements, 3], order = 'C')
    labels_array= np.ravel(labels, order='F')
    
    # remove elements, nodes and labels outside sphere
    idx_elements_inside = np.nonzero(labels_array)
    labels_array = labels_array[labels_array != 0] - 1
    elements_inside = np.squeeze(elements[idx_elements_inside,:])
    idx_nodes_inside = np.unique(np.ravel(elements_inside))
    palette = np.unique(np.ravel(elements_inside))
    key = np.arange(idx_nodes_inside.size)
    index = np.digitize(elements_inside.ravel(), palette, right=True)
    elements_inside = key[index].reshape(elements_inside.shape)

    np.savez_compressed(path, elements=elements_inside.astype(int),nodes=nodes_array[idx_nodes_inside.astype(int),:],labels=labels_array,centers=centers_array[idx_elements_inside,:])
