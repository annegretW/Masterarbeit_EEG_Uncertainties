#!/usr/bin/python3

from webbrowser import get
import server
import eeg_model
import umbridge
import numpy as np
from utility_functions import saveStructuredHexMeshForSphere, save_leadfield_matrix, save_transfer_matrix, calc_disturbed_sensor_values, get_dipole

def create_leadfields():
    dims = [16, 32, 64, 128]

    path_leadfield_matrix = [
        "../data/leadfield_matrix_1005_16",
        "../data/leadfield_matrix_1005_32",
        "../data/leadfield_matrix_1005_64",
        "../data/leadfield_matrix_1005_128"]

    electrodes_path = "../data/electrodes_1005.txt"
    
    for i in range(len(dims)):
        save_leadfield_matrix(electrodes_path, path_leadfield_matrix[i], dims[i])

#create_leadfields()
#N = 128
#center = (127,127,127)
#radii = (92,86,80,78)
#path = "../data/mesh_128.npz"
#saveStructuredHexMeshForSphere(N, center, radii, path)

# parameter
center = (127,127,127)
radii = (92,86,80,78)
conductivities = [0.00043,0.00001,0.00179,0.00033]

# chosen reference source
point = [150,150,150]
s_ref = get_dipole(point)

# create a testmodel
b_1010, sigma_0 = calc_disturbed_sensor_values(s_ref, "../data/electrodes_1010.txt")

b_ref = [b_1010, b_1010, b_1010, b_1010]
print(b_1010)
sigma = [8*sigma_0, 6*sigma_0, 4*sigma_0, 2*sigma_0]

transfer_path_list = [
    "../data/transfer_matrix_1010_16.npz",
    "../data/transfer_matrix_1010_32.npz",
    "../data/transfer_matrix_1010_64.npz"]

leadfield_path_list = [
    "../data/leadfield_matrix_1010_16.npz",
    "../data/leadfield_matrix_1010_32.npz",
    "../data/leadfield_matrix_1010_64.npz",
    "../data/leadfield_matrix_1010_128.npz"]

mesh_path_list = [
    "../data/mesh_16.npz",
    "../data/mesh_32.npz",
    "../data/mesh_64.npz",
    "../data/mesh_128.npz"]

cells_per_dim = [16,32,64,128]

testmodel = eeg_model.EEGModelFromFile(
    b_ref, 
    sigma, 
    leadfield_path_list, 
    mesh_path_list, 
    cells_per_dim,
    center, 
    radii, 
    conductivities)

# send via localhost
umbridge.serve_model(testmodel, 4243) 

'''
path_mesh = "../data/mesh_128"
path_transfer_matrix = "../data/transfer_matrix_1005_128"
electrodes_path = "../data/electrodes_1005.txt"

save_transfer_matrix(path_mesh, electrodes_path, path_transfer_matrix, 128)
'''

'''
center = (127,127,127)
radii = (92,86,80,78)

path_mesh = "../data/mesh_64"

saveStructuredHexMeshForSphere(64,center,radii,path_mesh)
'''

