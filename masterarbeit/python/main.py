#!/usr/bin/python3

import server
import eeg_model
import umbridge
import numpy as np
from utility_functions import save_leadfield_matrix, calc_disturbed_sensor_values

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

mesh_path='../data/tet_mesh.msh'
tensors_path='../data/conductivities.txt'
electrodes_path='../data/electrodes.txt'

# define different test models
def testmodel_example1():
    return server.TestModel('/home/anne/Masterarbeit/masterarbeit/data/leadfield_matrix_10')

def testmodel_example2():
    return server.TestModel('/home/anne/Masterarbeit/masterarbeit/data/leadfield_matrix_100')

def testmodel_example3(b):
    return eeg_model.EEGModelNew(b)

def testmodel_example4(b):
    return eeg_model.EEGModelTransferFromFile(b)

# one example of b_ref to save computation time
b_ref = [[-1.44163533, -1.76093881,  5.27508808, -2.6786638,  -1.3064591,  -2.27850429,
  -2.07960639, -1.24571309, 12.20550839, -2.6565059,  -2.29963664 ,-2.46565563,
  -2.71329052, -1.4184369,  -0.09620508, -2.2908153 , -2.14041701, 12.20740432,
  -1.20069863, -2.09782968, -2.51178363, -1.07017265,-2.51989558,  0.53272719,
  10.44195475, -2.46205758, -2.52244736, -2.50416979 ,-2.12300088, -2.20787697,
  -1.37599416, -2.66457723, -2.67881998, -1.44955414 ,-2.04038605,  0.93386173,
   0.11946625, -2.30158351,  5.15885421, -0.39175818 ,-2.59070825 , 1.07069382,
   3.71325134,  0.53624818 ,-0.25503899, -1.79731908 ,-1.33850562 , 0.58348098,
  -1.75514711,  0.52679633,  0.53349739, -2.68844983,  2.86481816, -2.11302868,
   5.18029592, -1.36527411 ,-2.47462783, -0.39910639, -2.55367746,  4.86883174,
  -2.1913921,  -2.10747477,  3.46738841 ,-2.3445429 ,  0.1203936 , -2.55334512,
  -1.43177645, 14.6304374 ,  4.58796473,  3.38983978]]

# chosen reference source
s_ref = dp.Dipole3d([127, 127, 190, 0, 0, 1])

# choose a testmodel
#testmodel = testmodel_example3(calc_disturbed_sensor_values(s_ref))
testmodel = testmodel_example4(calc_disturbed_sensor_values(s_ref))

# send via localhost
umbridge.serve_model(testmodel, 4243) 

#path_mesh = "../data/mesh_32"
#path_leadfield = "../data/leadfield_32"

#save_leadfield_matrix(path_mesh, path_leadfield, 32)

#path_mesh = "../data/mesh_128"
#path_transfer_matrix = "../data/transfer_matrix_128"

#save_leadfield_matrix(path_mesh, path_transfer_matrix, 128)

