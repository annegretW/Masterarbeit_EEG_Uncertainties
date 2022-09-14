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
cells_per_dim = 16
cell_size = 2*np.max(radii)/cells_per_dim

# create mesh
mesh = msh.StructuredMesh(center, radii, cells_per_dim)

print(mesh.nodes[110:140])
