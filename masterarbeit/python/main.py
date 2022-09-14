#!/usr/bin/python3

from webbrowser import get
import server
import eeg_model
import umbridge
import numpy as np
from utility_functions import saveStructuredHexMeshForSphere, save_leadfield_matrix, save_transfer_matrix, calc_disturbed_sensor_values, get_dipole

def create_leadfields():
    dims = [16, 32, 64]

    path_leadfield_matrix = [
        "../data/leadfield_matrix_1005_16",
        "../data/leadfield_matrix_1005_32",
        "../data/leadfield_matrix_1005_64"]

    electrodes_path = "../data/electrodes_1005.txt"
    
    for i in len(dims):
        save_leadfield_matrix(electrodes_path, path_leadfield_matrix[i], dims[i])


# parameter
center = (127,127,127)
radii = (92,86,80,78)
conductivities = [0.00043,0.00001,0.00179,0.00033]

# chosen reference source
point = [110, 120, 130]
s_ref = get_dipole(point)

# create a testmodel
b_1005, sigma = calc_disturbed_sensor_values(s_ref, "../data/electrodes_1005.txt")

b_ref = [b_1005, b_1005, b_1005]

transfer_path_list = [
    "../data/transfer_matrix_1005_16.npz",
    "../data/transfer_matrix_1005_32.npz",
    "../data/transfer_matrix_1005_64.npz"]

leadfield_path_list = [
    "../data/leadfield_matrix_1005_16.npz",
    "../data/leadfield_matrix_1005_32.npz",
    "../data/leadfield_matrix_1005_64.npz"]

mesh_path_list = [
    "../data/mesh_16.npz", 
    "../data/mesh_32.npz",
    "../data/mesh_64.npz"]

cells_per_dim = [16,32,64]

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

