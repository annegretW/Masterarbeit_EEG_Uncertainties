import h5py
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import server 

fin = h5py.File('/home/anne/Masterarbeit/masterarbeit/results/samples_mlda.h5')
samples_l0 = np.array( fin['/samples'] )

testmodel = server.TestModel()

theta = np.transpose(np.array(samples_l0))

#print(testmodel.posterior(theta[400]))

#for i in range(900):
#    b = np.matmul(testmodel.leadfield, theta[i])
#
#    print(np.linalg.norm(b-testmodel.b_ref))

print(testmodel.b_ref)

s_ref = np.ones(1000)
s_ref = 1*s_ref

print(np.matmul(testmodel.leadfield,s_ref))
