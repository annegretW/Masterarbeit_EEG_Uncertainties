#!/usr/bin/python3

from webbrowser import get
import server
import eeg_model
import umbridge
import numpy as np
from utility_functions import save_leadfield_matrix, save_transfer_matrix, calc_disturbed_sensor_values, get_dipole

# chosen reference source
point = [100, 130, 160]
s_ref = get_dipole(point)
#print(s_ref)

# create a testmodel
#b_1020 = calc_disturbed_sensor_values(s_ref, "../data/electrodes_1020.txt")
#b_1010 = calc_disturbed_sensor_values(s_ref, "../data/electrodes_1010.txt")
#b_1005 = calc_disturbed_sensor_values(s_ref, "../data/electrodes_1005.txt")
b = calc_disturbed_sensor_values(s_ref, "../data/electrodes.txt")

b_ref = [b, b, b]
testmodel = eeg_model.EEGModelTransferFromFile(b_ref)

# send via localhost
umbridge.serve_model(testmodel, 4243) 

#path_mesh = "../data/mesh_32"
#path_leadfield = "../data/leadfield_32"

#save_leadfield_matrix(path_mesh, path_leadfield, 32)

#path_mesh = "../data/mesh_16"
#path_transfer_matrix = "../data/transfer_matrix_16"
#electrodes_path = "../data/electrodes.txt"

#save_transfer_matrix(path_mesh, electrodes_path, path_transfer_matrix, 16)

