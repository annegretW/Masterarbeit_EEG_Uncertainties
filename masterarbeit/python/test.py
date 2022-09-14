from operator import truediv
import numpy as np
import structured_mesh as msh
from leadfield import create_leadfield, create_transfer_matrix
import transfer_matrix  
import utility_functions
from structured_mesh import find_next_center

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

# Calc correct sensor values
mesh_path='../data/tet_mesh.msh'
tensors_path='../data/conductivities.txt'
electrodes_path='../data/electrodes_1010.txt'

conductivities = [0.00043,0.00001,0.00179,0.00033]

point = [114.0625, 102.5625,  39.3125]
point1 = [125.34709414, 146.43341941, 152.30150197]
point2 = [124.93838629, 103.32541128, 110.73780151]

# create mesh
mesh = np.load('../data/mesh_64.npz')
center = (127,127,127)
radii = (92,86,80,78)
cells_per_dim = 64

s_ref = utility_functions.get_dipole(point)
print(s_ref)
s_val0 = utility_functions.get_dipole(find_next_center(center, radii, cells_per_dim, point))
print(s_val0)
s_val1 = utility_functions.get_dipole(find_next_center(center, radii, cells_per_dim, point1))
print(s_val1)
s_val2 = utility_functions.get_dipole(find_next_center(center, radii, cells_per_dim, point2))
print(s_val2)

b_ref = utility_functions.calc_disturbed_sensor_values(s_ref, "../data/electrodes_1005.txt")[0]
print(b_ref)

solver_cfg = {
    'reduction' : '1e-14', 
    'edge_norm_type' : 'houston', 
    'penalty' : '20', 
    'scheme' : 'sipg', 
    'weights' : 'tensorOnly'}

source_model_cfg = {
        'type' : 'venant',
        'numberOfMoments' : 3,
        'referenceLength' : 20,
        'weightingExponent' : 1,
        'relaxationFactor' : 1e-6,
        'mixedMoments' : True,
        'restrict' : True,
        'initialization' : 'closest_vertex'
        }

#T2 = np.load("../data/transfer_matrix_1010_64.npz")['arr_0']

config = {
    'solver.reduction' : 1e-10,
    'source_model' : source_model_cfg,
    'post_process' : True,
    'subtract_mean' : True
}

elements = mesh['elements']
nodes = mesh['nodes']
labels = mesh['labels']

meg_config = {
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
      },
     'post_process' : 'true', 
    'subtract_mean' : 'true'
 }
meg_driver = dp.MEEGDriver3d(meg_config)

'''
b_val0 = meg_driver.applyEEGTransfer(T2,[s_val0],config)[0]
b_val1 = meg_driver.applyEEGTransfer(T2,[s_val1],config)[0]
b_val2 = meg_driver.applyEEGTransfer(T2,[s_val2],config)[0]
'''

L = np.transpose(np.load("../data/leadfield_matrix_1005_64.npz")['arr_0'])

index = msh.find_next_center_index(mesh['centers'][0], point)
print(index)
b_val0 = L[index]
print(mesh['centers'][0][index])
index = msh.find_next_center_index(mesh['centers'][0], point1)
print(index)
b_val1 = L[index]
print(mesh['centers'][0][index])
index = msh.find_next_center_index(mesh['centers'][0], point2)
print(index)
b_val2 = L[index]
print(mesh['centers'][0][index])

print(np.amax(np.array(b_ref)-np.array(b_val0)))
print(np.linalg.norm(np.array(b_ref)-np.array(b_val0), 2))
print("-------------------------------------------------------------")
print(np.amax(np.array(b_ref)-np.array(b_val1)))
print(np.linalg.norm(np.array(b_ref)-np.array(b_val1), 2))
print("-------------------------------------------------------------")
print(np.amax(np.array(b_ref)-np.array(b_val2)))
print(np.linalg.norm(np.array(b_ref)-np.array(b_val2), 2))
