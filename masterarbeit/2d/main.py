import sys
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
        "data/mesh_20_1.msh",
        "data/mesh_20_2.msh",
        "data/mesh_20_3.msh"]

l = len(path_leadfield_matrices)
assert l == len(path_meshs)

# Set parameters
center = [127,127]
radii = [92,86,78]
conductivities = [0.00043,0.00001,0.00179]

# Set mode ('Radial dipole','Arbitrary dipole orientation')
mode = 'Arbitrary dipole orientation'
#mode = 'Radial dipole'

# Set dipole
position = [80, 150]
rho = 1

if mode == 'Radial dipole':
    s_ref = utility_functions.get_radial_dipole(position,center)
else:
    s_ref = utility_functions.get_dipole(position,center,rho)

# Set noise
relative_noise = 0.01

# Set variance factor for each level
var_factor = [8, 2, 1]

# model (L - Use leadfield, T - Use transfer matrix)
model = 'T'

# Generate electrode positions if not already existing
if not exists(path_electrodes):
    utility_functions.get_electrodes(path_meshs[0])

m = len(np.load(path_electrodes)["arr_0"])
print("Number of electrodes: " + str(m))

# Create leadfield matrices if not already existing
if model=='L':
    for i in range(l):
        if not exists(path_leadfield_matrices[i]):
            utility_functions.save_leadfield_matrix(
                path_electrodes, 
                path_conductivities, 
                path_meshs[i], 
                path_leadfield_matrices[i])

# Create transfer matrices if not already existing
elif model == 'T':
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

if relative_noise==0:
    b_ref[0] = utility_functions.calc_sensor_values(s_ref, path_electrodes, path_meshs[-1], path_conductivities)
    sigma_0 = 0.001*np.amax(np.absolute(b_ref[0]))
else:
    b_ref[0], sigma_0 = utility_functions.calc_disturbed_sensor_values(s_ref, path_electrodes, path_meshs[-1], path_conductivities, relative_noise)

sigma[0] = var_factor[0]*sigma_0

for i in range(1,l):
    b_ref[i] = b_ref[0]
    sigma[i] = var_factor[i]*sigma_0 

# Create EEG modell
if model=='T':
    testmodel = eeg_model.EEGModelTransferFromFile(b_ref, sigma, path_transfer_matrices, path_meshs, path_conductivities, center, mode)
elif model == 'L':
    testmodel = eeg_model.EEGModelFromFile(b_ref, sigma, path_leadfield_matrices, path_meshs)

# send via localhost
serve_model(testmodel, 4243)