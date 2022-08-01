#!/usr/bin/python3

import umbridge
import numpy as np
import math

class TestModel(umbridge.Model):
    def __init__(self, leadfield_path):
        super(TestModel, self).__init__()
        L = np.transpose(np.load(leadfield_path + '_L1.npz')['arr_0'])
        self.n = len(L[0])

        # create reference source coefficients
        # self.s_ref = np.random.rand(self.n)
        self.s_ref = np.zeros(self.n)
        self.s_ref[0] = 1

        self.leadfield1 = L 
        self.m1 = len(L) 
        self.b_ref1 = np.matmul(self.leadfield1,self.s_ref)
        print(self.b_ref1)

        L = np.transpose(np.load(leadfield_path + '_L2.npz')['arr_0'])
        self.leadfield2 = L 
        self.m2 = len(L) 
        self.b_ref2 = np.matmul(self.leadfield2,self.s_ref)

        L = np.transpose(np.load(leadfield_path + '_L3.npz')['arr_0'])
        self.leadfield3 = L 
        self.m3 = len(L) 
        self.b_ref3 = np.matmul(self.leadfield3,self.s_ref)

    def get_input_sizes(self):
        return [self.n]

    def get_output_sizes(self):
        return [1]

    def posterior(self, theta, level):
        sigma = 0.1

        if level==1:
            b = np.matmul(self.leadfield1, theta)
            posterior = ((1/(2*math.pi*sigma**2))**(self.m1/2))*(math.exp(-(1/(2*sigma**2))*(np.linalg.norm(self.b_ref1-b))**2))
        if level==2:
            b = np.matmul(self.leadfield2, theta)
            posterior = ((1/(2*math.pi*sigma**2))**(self.m2/2))*(math.exp(-(1/(2*sigma**2))*(np.linalg.norm(self.b_ref2-b))**2))
        if level==3:
            b = np.matmul(self.leadfield3, theta)
            posterior = ((1/(2*math.pi*sigma**2))**(self.m3/2))*(math.exp(-(1/(2*sigma**2))*(np.linalg.norm(self.b_ref3-b))**2))
        
        if(posterior <= 0): 
            print(posterior)
            return 0
        else:
            return math.log(posterior)

    def __call__(self, parameters, config={'level': 1}):
        return [[self.posterior(parameters[0],config["level"])]]

    def supports_evaluate(self):
        return True