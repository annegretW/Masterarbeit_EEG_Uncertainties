#!/usr/bin/python3

import server
import umbridge
import numpy as np


testmodel = server.TestModel()
print(testmodel.s_ref[0])
print(testmodel.s_ref[1])

print(testmodel.n)
print(testmodel.m1)
print(testmodel.m2)
print(testmodel.m3)


#theta = np.zeros(1000)
#theta[0] = 1
#print(testmodel.posterior(theta))

# send via localhost
umbridge.serve_model(testmodel, 4243) 
