import numpy as np
import structured_mesh as msh
from leadfield import create_leadfield
import h5py

conductivities = [0.00043,0.00001,0.00179,0.00033]
center = (127,127,127)
radii = (92,86,80,78)
cells_per_dim = 64
cell_size = 2*np.max(radii)/cells_per_dim

point = np.array([127,127,190])

# create mesh
mesh = msh.StructuredMesh(center, radii, cells_per_dim)
print(mesh.centers)

print(mesh.find_next_center(point))


#next = n[0] + (cells_per_dim+1)*n[1] + (cells_per_dim+1)**2*n[2]

#fin = h5py.File('/home/anne/Masterarbeit/masterarbeit/results/test.h5')
#samples = np.array( fin['/samples'] )
#fin.close()

#print(samples)