#!/usr/bin/python3

import umbridge
import numpy as np
from scipy import stats
import utility_functions
import meshio
import math

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

#################################################################################################
# This is an EEG model, containing lists of meshes and the according leadfield                  #
# matrices - one per level                                                                      #   
#                                                                                               #
# Attributes:                                                                                   #
#   mesh                - list of meshs of different coarseness (one per level)                 #
#   leadfield_matrix    - list of ladfield matrices according to the meshs (one per level)      #   
#   b_ref               - reference values of sources                                           #
#   sigma               - posterior variance                                                    #
##################################################################################################
class EEGModelFromFile(umbridge.Model):
    def __init__(self, b_ref, sigma, leadfield_path_list, mesh_path_list):
        super(EEGModelFromFile, self).__init__()

        # initialize lists
        self.mesh = []
        self.leadfield_matrix = []

        for i in range(len(mesh_path_list)):
            # read mesh
            mesh = meshio.read(mesh_path_list[i])
            self.mesh.append(mesh)
            #self.mesh.append(msh.StructuredMesh(center, radii, cells_per_dim[i]))
            self.leadfield_matrix.append(np.load(leadfield_path_list[i])['arr_0'])

        # reference values of measurement values
        self.b_ref = b_ref
        self.m = len(b_ref[0])

        # posterior variance
        self.sigma = sigma


    def get_input_sizes(self):
        return [3]

    def get_output_sizes(self):
        return [1]

    # Calculates the posterior probability of the source theta on a given level
    def posterior(self, theta, level):
        #print(level)
        #print(theta)
        i = level-1

        if (math.sqrt((theta[0]-127)**2+(theta[1]-127)**2)>78):
            #print(theta)
            return -1e20

        # find next node to theta on the mesh and select the according leadfield
        points = np.array(self.mesh[i].points[:,0:2])
        index = utility_functions.find_next_node(points,theta[0:2])
        #index = self.mesh[i].find_next_center_index(theta)
        b = np.array(self.leadfield_matrix[i][index])
        #b -= b[0]
        #print(b)
        #node = self.mesh[i].nodes[i]

        # calculate the posterior as a normal distribution
        #error = utility_functions.relative_error(self.b_ref, b)
        #error = np.amax(np.absolute(np.array(self.b_ref)-np.array(b)))
        #posterior = -(1/(2*self.sigma[i]**2))*error**2
        #posterior = stats.multivariate_normal.pdf(b,mean=self.b_ref[i],cov=self.sigma[i]*np.identity(self.m))
        #print(posterior)
        #if posterior!=0:
        #    posterior = np.log(posterior)
        #print(posterior)
        posterior = ((1/(2*self.sigma[i]**2))**(self.m/2))*np.exp(-(1/(2*self.sigma[i]**2))*(np.linalg.norm(np.array(self.b_ref[i])-np.array(b), 2)/np.linalg.norm(np.array(self.b_ref[i]), 2))**2)
        if posterior==0:
           return -1e20
        
        return np.log(posterior)
        
    def __call__(self, parameters, config={'level': 1}):
        return [[self.posterior(parameters[0],config["level"])]]

    def supports_evaluate(self):
        return True


#################################################################################################
# This is an EEG model, containing lists of meshes and the according transfer                   #
# matrices - one per level                                                                      #   
#                                                                                               #
# Attributes:                                                                                   #
#   mesh                - list of meshs of different coarseness (one per level)                 #
#   transfer_matrix     - list of transfer matrices according to the meshs (one per level)      #   
#   b_ref               - reference values of sources                                           #
#   sigma               - posterior variance                                                    #
##################################################################################################
class EEGModelTransferFromFile(umbridge.Model):
    def __init__(self, b_ref, sigma, transfer_path_list, mesh_path_list, conductivities_path, center):
        super(EEGModelTransferFromFile, self).__init__()

        # initialize lists
        self.mesh = []
        self.transfer_matrix = []
        self.meg_drivers = []

        for i in range(len(mesh_path_list)):
            # read mesh
            mesh = meshio.read(mesh_path_list[i])
            self.mesh.append(mesh)
            print(mesh.points)
            #self.mesh.append(msh.StructuredMesh(center, radii, cells_per_dim[i]))
            self.transfer_matrix.append(np.load(transfer_path_list[i])['arr_0'])

            config = {
                'type' : 'fitted',
                'solver_type' : 'cg',
                'element_type' : 'tetrahedron',
                'volume_conductor': {
                    'grid.filename' : mesh_path_list[i], 
                    'tensors.filename' : conductivities_path
                },
                'post_process' : 'true', 
                'subtract_mean' : 'true'
            }
            meg_driver = dp.MEEGDriver2d(config)
            self.meg_drivers.append(meg_driver)

        # reference values of measurement values
        self.b_ref = b_ref
        self.m = len(b_ref[0])
        
        # posterior variance
        self.sigma = sigma

        self.center = center
        
        source_model_config = {
            'type' : 'venant',
            'numberOfMoments' : 3,
            'referenceLength' : 20,
            'weightingExponent' : 1,
            'relaxationFactor' : 1e-6,
            'mixedMoments' : True,
            'restrict' : True,
            'initialization' : 'closest_vertex'
        }

        self.config = {
            'solver.reduction' : 1e-10,
            'source_model' : source_model_config,
            'post_process' : True,
            'subtract_mean' : True
        }


    def get_input_sizes(self):
        return [3]

    def get_output_sizes(self):
        return [1]

    # Calculates the posterior probability of the source theta on a given level
    def posterior(self, theta, level):
        i = level-1

        if (math.sqrt((theta[0]-127)**2+(theta[1]-127)**2)>78):
            return -1e20

        next_dipole = utility_functions.get_dipole(theta[0:2], self.center, theta[2])
        b = self.meg_drivers[i].applyEEGTransfer(self.transfer_matrix[i],[next_dipole],self.config)[0]

        # calculate the posterior as a normal distribution
        posterior = ((1/(2*self.sigma[i]**2))**(self.m/2))*np.exp(-(1/(2*self.sigma[i]**2))*(np.linalg.norm(np.array(self.b_ref[i])-np.array(b), 2)/np.linalg.norm(np.array(self.b_ref[i]), 2))**2)
        if posterior==0:
           return -1e20
        
        return np.log(posterior)
        
    def __call__(self, parameters, config={'level': 1}):
        return [[self.posterior(parameters[0],config["level"])]]

    def supports_evaluate(self):
        return True