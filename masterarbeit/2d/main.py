import numpy as np

from os.path import exists
from umbridge import serve_model

import utility_functions
import eeg_model

# Set paths
path_electrodes = "data/electrodes.npz"

path_conductivities = 'data/conductivities.txt'

path_leadfield_matrices = [
        "data/leadfield_matrix_1.npz",
        "data/leadfield_matrix_2.npz",
        "data/leadfield_matrix_3.npz"]

path_transfer_matrices = [
        "data/transfer_matrix_1.npz",
        "data/transfer_matrix_2.npz",
        "data/transfer_matrix_3.npz"]

path_meshs = [
        "data/mesh_30_1.msh",
        "data/mesh_30_2.msh",
        "data/mesh_30_3.msh"]

l = len(path_leadfield_matrices)
assert l == len(path_meshs)

# Set parameters
center = [127,127]
radii = [92,86,78]
conductivities = [0.00043,0.00001,0.00179]

# Set dipole
position = [119.7987347,  90.18871387]
s_ref = utility_functions.get_radial_dipole(position,center)

# Set noise
relative_noise = 0.001

# Set variance factor for each level
var_factor = [1, 1, 1]

# Mode (L - Use leadfield, T - Use transfer matrix)
mode = 'T'

# Generate electrode positions if not already existing
if not exists(path_electrodes):
    utility_functions.get_electrodes(path_meshs[0])

m = len(np.load(path_electrodes)["arr_0"])
print("Number of electrodes: " + str(m))

# Create leadfield matrices if not already existing
if mode=='L':
    for i in range(l):
        if not exists(path_leadfield_matrices[i]):
            utility_functions.save_leadfield_matrix(
                path_electrodes, 
                path_conductivities, 
                path_meshs[i], 
                path_leadfield_matrices[i])

# Create transfer matrices if not already existing
elif mode == 'T':
    for i in range(l):
        if not exists(path_transfer_matrices[i]):
            utility_functions.save_transfer_matrix(
                path_electrodes, 
                path_conductivities, 
                path_meshs[i], 
                path_transfer_matrices[i])

# Generate reference values
b_ref = np.zeros((l,m))
sigma = np.zeros(l)

b_ref[0], sigma_0 = utility_functions.calc_disturbed_sensor_values(s_ref, path_electrodes, path_meshs[-1], path_conductivities, relative_noise)
sigma[0] = var_factor[0]*sigma_0

for i in range(1,l):
    b_ref[i] = b_ref[0]
    sigma[i] = var_factor[i]*sigma_0 

# Create EEG Model
if mode=='T':
    testmodel = eeg_model.EEGModelTransferFromFile(b_ref, sigma, path_transfer_matrices, path_meshs, path_conductivities, center)
elif mode == 'L':
    testmodel = eeg_model.EEGModelFromFile(b_ref, sigma, path_leadfield_matrices, path_meshs)

# send via localhost
serve_model(testmodel, 4243)