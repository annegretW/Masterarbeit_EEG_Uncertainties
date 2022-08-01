#!/usr/bin/python3

import server
import umbridge
import numpy as np

# define different test models
testmodel_example1 = server.TestModel('/home/anne/Masterarbeit/masterarbeit/data/leadfield_matrix_10')
testmodel_example2 = server.TestModel('/home/anne/Masterarbeit/masterarbeit/data/leadfield_matrix_100')

# choose a testmodel
testmodel = testmodel_example2

# print information
print("n = " + str(testmodel.n))
print("s[0] = " + str(testmodel.s_ref[0]))
print("s[1] = " + str(testmodel.s_ref[1]))

# send via localhost
umbridge.serve_model(testmodel, 4243) 
