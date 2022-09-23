#!/usr/bin/python3

import umbridge
import numpy as np
from leadfield import create_leadfield, create_transfer_matrix
import structured_mesh as msh
import utility_functions
import meshio

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
            print(i)
            self.leadfield_matrix.append(np.load(leadfield_path_list[i])['arr_0'])

        # reference values of measurement values
        self.b_ref = b_ref
        '''for b in b_ref:
            b= np.array(b)
            b -= b[0]
            self.b_ref.append(b)
        print(self.b_ref)'''

        print(self.b_ref)
        # posterior variance
        self.sigma = sigma


    def get_input_sizes(self):
        return [2]

    def get_output_sizes(self):
        return [1]

    # Calculates the posterior probability of the source theta on a given level
    def posterior(self, theta, level):
        i = level-1

        # find next node to theta on the mesh and select the according leadfield
        points = np.array(self.mesh[i].points[:,0:2])
        index = utility_functions.find_next_node(points,theta)
        #index = self.mesh[i].find_next_center_index(theta)
        #print(index)
        b = np.array(self.leadfield_matrix[i][index])
        #b -= b[0]
        #print(b)
        #node = self.mesh[i].nodes[i]

        # calculate the posterior as a normal distribution
        #error = utility_functions.relative_error(self.b_ref, b)
        #error = np.amax(np.absolute(np.array(self.b_ref)-np.array(b)))
        #posterior = -(1/(2*self.sigma[i]**2))*error**2
        posterior = -(1/(2*self.sigma[i]**2))*(np.linalg.norm(np.array(self.b_ref[i])-np.array(b), 2))**2

        return posterior
        
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
    def __init__(self, b_ref, sigma_0, transfer_path_list, mesh_path_list, cells_per_dim, center, radii, conductivities):
        super(EEGModelTransferFromFile, self).__init__()

        # parameters
        sigma = [10*sigma_0,5*sigma_0,sigma_0]

        # initialize lists
        self.mesh = []
        self.transfer_matrix = []
        self.meg_drivers = []

        for i in range(len(cells_per_dim)):
            # create mesh
            self.mesh.append(np.load(mesh_path_list[i], allow_pickle=True))
            #self.mesh.append(msh.StructuredMesh(center, radii, cells_per_dim[i]))
            self.transfer_matrix.append(np.load(transfer_path_list[i])['arr_0'])

            meg_config = {
                'type' : 'fitted',
                'solver_type' : 'cg',
                'element_type' : 'hexahedron',
                'volume_conductor' : {
                    'grid' : {
                        'elements' : self.mesh[i]['elements'].tolist(),
                        'nodes' : self.mesh[i]['nodes'].tolist()
                    },
                    'tensors' : {
                        'labels' : self.mesh[i]['labels'].tolist(),
                        'conductivities' : conductivities
                    }
                },
                'post_process' : 'true', 
                'subtract_mean' : 'true'
            }
            meg_driver = dp.MEEGDriver3d(meg_config)
            self.meg_drivers.append(meg_driver)

        # reference values of measurement values
        self.b_ref = b_ref
        
        # posterior variance
        self.sigma = sigma
        self.center = center
        self.radii = radii
        self.cells_per_dim = cells_per_dim
        
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

        next_dipole_pos = msh.find_next_center(self.center, self.radii, self.cells_per_dim[i], theta) 
        #next_dipole_pos = theta
        dist = np.sqrt(sum([(x-c)**2 for x,c in zip(next_dipole_pos, self.center)]))

        if dist > np.max(self.radii):
            return 0

        next_dipole = utility_functions.get_dipole(next_dipole_pos)
        b = self.meg_drivers[i].applyEEGTransfer(self.transfer_matrix[i],[next_dipole],self.config)[0]

        # calculate the posterior as a normal distribution
        posterior = -(1/(2*self.sigma[i]**2))*(np.linalg.norm(np.array(self.b_ref[i])-np.array(b), 2))**2

        #print([next_dipole_pos[0],next_dipole_pos[1],next_dipole_pos[2]])
        #return [[posterior],[next_dipole_pos[0],next_dipole_pos[1],next_dipole_pos[2]]]
        #print(np.linalg.norm(np.array(self.b_ref[i])-np.array(b), 2))
        return posterior
        
    def __call__(self, parameters, config={'level': 1}):
        return [[self.posterior(parameters[0],config["level"])]]

    def supports_evaluate(self):
        return True