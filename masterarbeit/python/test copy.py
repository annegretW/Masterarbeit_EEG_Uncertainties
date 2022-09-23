from operator import truediv
import numpy as np
import structured_mesh as msh
from leadfield import create_leadfield, create_transfer_matrix
import transfer_matrix  
import utility_functions

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

conductivities = [0.00043,0.00001,0.00179,0.00033]
center = (127,127,127)
radii = (92,86,80,78)
cells_per_dim = 32
cell_size = 2*np.max(radii)/cells_per_dim

# create mesh
#mesh = np.load('../data/mesh_32.npz')

#for c in mesh['nodes']:
#    print(c)
'''
electrodes_path = "../data/electrodes_1005.txt"
path_leadfield_matrix = "../data/leadfield_matrix_test"
dim = 32

utility_functions.save_leadfield_matrix(electrodes_path, path_leadfield_matrix, dim)
'''

point1 = [110,120,130]
point2 = [130,130,130]
point3 = [150,150,150]
point4 = [160,160,160]

points = [point1, point2, point3, point4]

# create mesh
center = (127,127,127)
radii = (92,86,80,78)
cells_per_dim = [16,32,64,128]

for p in points:
    print("____________________________________________")
    print(p)
    print("____________________________________________")
    for d in cells_per_dim:
        print(msh.find_next_center(center,radii,d,p))