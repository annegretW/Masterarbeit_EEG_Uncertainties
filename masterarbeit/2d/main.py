#!/usr/bin/python3

from stat import UF_APPEND
from webbrowser import get
import eeg_model
import umbridge
import numpy as np
import utility_functions as uf 

def create_leadfields():

    path_leadfield_matrix = [
        "data/leadfield_matrix_1.npz",
        "data/leadfield_matrix_2.npz",
        "data/leadfield_matrix_3.npz"]

    path_meshs = [
        "data/mesh_1.msh",
        "data/mesh_2.msh",
        "data/mesh_3.msh"]

    electrodes_path = "data/electrodes.npz"
    
    for i in range(len(path_leadfield_matrix)):
        uf.save_leadfield_matrix(electrodes_path, path_meshs[i], path_leadfield_matrix[i])

import meshio

#create_leadfields()

# chosen reference source
point = [150,150]
s_ref = uf.get_dipole(point,[127,127])

# create a testmodel
b, sigma_0 = uf.calc_disturbed_sensor_values(s_ref, "data/electrodes.npz")

b_ref = [b, b, b]

sigma = [4*sigma_0, 2*sigma_0, sigma_0]

leadfield_path_list = [
        "data/leadfield_matrix_1.npz",
        "data/leadfield_matrix_2.npz",
        "data/leadfield_matrix_3.npz"]

mesh_path_list = [
        "data/mesh_1.msh",
        "data/mesh_2.msh",
        "data/mesh_3.msh"]

testmodel = eeg_model.EEGModelFromFile(
    b_ref, 
    sigma, 
    leadfield_path_list, 
    mesh_path_list)

# send via localhost
umbridge.serve_model(testmodel, 4243) 