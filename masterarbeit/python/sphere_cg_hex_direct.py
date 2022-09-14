# Compute EEG leadfield using the standard (CG-) FEM approach,
# in a 4-layer spherical hexahedral sphere model
# with the St. Venant source model
# by solving the system directly for a test dipole

# I. Import libraries
import numpy as np
import utility_functions

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp
import os

# II. Define input files
folder_input = '../data/'
folder_output = '../results/'
#filename_dipoles = os.path.join(folder_input, 'sphere_dipoles.txt')
filename_electrodes = os.path.join(folder_input, 'electrodes_1020.txt')
center = (127,127,127)
radii = (92,86,80,78)
conductivities = [0.00043,0.00001,0.00179,0.00033]

def cellsFromMM(mm):
    return int(np.round(2*np.max(radii)/mm))
def generateStructuredHexMeshForSphere(N):
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
    # create tissue labels for elements in bounding box
    centers = np.meshgrid(*[ll+(ur-ll)*(np.linspace(1/N,1,N)-.5/N) for ll,ur in zip(lower_left, upper_right)], indexing = 'ij')
    dist = np.sqrt(sum([(x-c)**2 for x,c in zip(centers, center)]))
    labels = np.zeros([N]*len(lower_left), dtype=int)
    for i, r in enumerate(sorted(radii, reverse=True)):
        labels[dist<r] = i+1
    labels_array= np.ravel(labels, order='F')
    # remove elements, nodes and labels outside sphere
    idx_elements_inside = np.nonzero(labels_array);
    labels_array = labels_array[labels_array != 0] - 1;
    elements_inside = np.squeeze(elements[idx_elements_inside,:])
    idx_nodes_inside = np.unique(np.ravel(elements_inside))
    palette = np.unique(np.ravel(elements_inside))
    key = np.arange(idx_nodes_inside.size)
    index = np.digitize(elements_inside.ravel(), palette, right=True)
    elements_inside = key[index].reshape(elements_inside.shape)
    return [elements_inside.astype(int),nodes_array[idx_nodes_inside.astype(int),:], labels_array]

[elements, nodes, labels] = generateStructuredHexMeshForSphere(8)
np.savez_compressed("../data/mesh_128", elements=elements, nodes=nodes, labels=labels)

# III. Create MEEG driver
# We create the driver object which will read the mesh along with the conductivity tensors from the provided files
config = {
    'type' : 'fitted',
    'solver_type' : 'cg',
    'element_type' : 'hexahedron',
    'volume_conductor' : {
        'grid' : {
            'elements' : elements.tolist(),
            'nodes' : nodes.tolist()
        },
        'tensors' : {
            'labels' : labels.tolist(),
            'conductivities' : conductivities
        }
    }
}
driver = dp.MEEGDriver3d(config)

# IV. Read and set electrode positions
# When projecting the electrodes, we choose the closest nodes
electrodes = np.genfromtxt(filename_electrodes,delimiter=None) 
electrodes = [dp.FieldVector3D(t) for t in electrodes.tolist()]
electrode_config = {
    'type' : 'closest_subentity_center',
    'codims' : [3]
}
driver.setElectrodes(electrodes, electrode_config)

# V. Compute transfer matrix
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

eeg_transfer_config = {
        'solver.reduction' : 1e-10,
        'source_model' : source_model_config,
        'post_process' : True,
        'subtract_mean' : True
    }

transfer_matrix, eeg_transfer_computation_information = driver.computeEEGTransferMatrix(eeg_transfer_config)

np.savez_compressed("../data/transfer_matrix_128_hex", transfer_matrix)

# V. Compute EEG leadfield
#Create source model configurations (St. Venant approach)post_processpost_process
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

# Read dipoles
dipoles = []
for n in nodes:
    dipoles.append(utility_functions.get_dipole(n))

# Compute leadfield for the first test dipole
x = driver.makeDomainFunction()
driver.solveEEGForward(dipoles, x, {
            'solver.reduction' : 1e-10,
            'source_model' : source_model_config,
            'post_process' : True,
            'subtract_mean' : True
        })
lf = driver.evaluateAtElectrodes(x)
print(lf)
lf -= np.mean(lf)
print(lf)
