import numpy as np
import structured_mesh as msh
from leadfield import create_leadfield

electrodes_path='../data/electrodes.txt'

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

# parameters
center = (127,127,127)
radii = (92,86,80,78)
conductivities2 = [0.00043,0.00001,0.00179,0.00033]
cells_per_dim = [8]
sigma = 0.005

mesh = msh.StructuredMesh(center, radii, cells_per_dim[0])
dipoles = []
for c in mesh.centers[0]:
    rho = np.arccos((c[2]-mesh.center[2])/np.max(mesh.radii))
    phi = np.arctan2(c[1]-mesh.center[1], c[0]-mesh.center[0])
    dipoles.append(dp.Dipole3d(c,[np.sin(phi)*np.cos(rho),np.sin(phi)*np.sin(rho),np.cos(phi)]))

# create transfer matrix and leadfield matrix
# T2, L2 = create_leadfield(mesh, conductivities2, electrodes_path, dipoles)

p = np.array([134, 156, 99])
p = p - np.array(center) + np.max(radii)
cell_size = 2*np.max(radii)/cells_per_dim[0]
print(cell_size)
p = np.divmod(p, cell_size)[0]
print(p)
print(p*cell_size + cell_size/2 + np.array(center) - np.max(radii) )