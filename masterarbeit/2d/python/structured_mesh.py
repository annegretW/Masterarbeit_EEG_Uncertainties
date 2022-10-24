import numpy as np
from scipy import integrate
from scipy.io import loadmat

class StructuredMesh():
    def __init__(self, coarseness, path):
        center = np.array([128, 128])
        width = 256
        cells_per_dim = int(256/coarseness)

        self.center = center
        self.cells_per_dim = cells_per_dim
        self.cell_size = width/self.cells_per_dim

        N = cells_per_dim
        nr_nodes = N + 1
        lower_left = center - width/2
        upper_right = center + width/2

        # create nodes array for bounding box
        nodes = np.meshgrid(*[np.linspace(ll,ur,num=N+1) for ll,ur in zip(lower_left, upper_right)], indexing='ij')
        nodes_array = np.reshape(np.transpose(np.array(nodes, order='F')),[nr_nodes**2, 2], order = 'C')
        
        # map elements to nodes in bounding box
        elements = np.empty(shape=(N**2,4))
        idx_nodes = np.arange(nodes_array.shape[0])
        idx_top =  idx_nodes[idx_nodes%nr_nodes**2  >= (nr_nodes)*(nr_nodes-1) ]
        idx_right = idx_nodes[idx_nodes%nr_nodes==nr_nodes-1]

        elements[:,0] = np.delete(idx_nodes, np.append(idx_right, idx_top))
        elements[:,1] = elements[:,0] + 1
        elements[:,[2,3]] = elements[:,[0,1]] + nr_nodes
        nr_elements = len(elements)

        # create tissue labels for elements in bounding box
        tissue_prob_map = loadmat('/home/anne/Masterarbeit/masterarbeit/2d//data/T1SliceAnne.mat')

        gray_prob = np.zeros((cells_per_dim, cells_per_dim))
        white_prob = np.zeros((cells_per_dim, cells_per_dim))
        skull_prob = np.zeros((cells_per_dim, cells_per_dim))
        csf_prob = np.zeros((cells_per_dim, cells_per_dim))
        scalp_prob = np.zeros((cells_per_dim, cells_per_dim))

        for i in range(cells_per_dim):
            for j in range(cells_per_dim):
                for k in range(coarseness):
                    for l in range(coarseness):
                        gray_prob[i,j] += tissue_prob_map['T1Slice']['gray'][0][0][coarseness*i+k,coarseness*j+l]
                        white_prob[i,j] += tissue_prob_map['T1Slice']['white'][0][0][coarseness*i+k,coarseness*j+l]
                        skull_prob[i,j] += tissue_prob_map['T1Slice']['skull'][0][0][coarseness*i+k,coarseness*j+l]
                        csf_prob[i,j] += tissue_prob_map['T1Slice']['csf'][0][0][coarseness*i+k,coarseness*j+l]
                        scalp_prob[i,j] += tissue_prob_map['T1Slice']['scalp'][0][0][coarseness*i+k,coarseness*j+l]
                gray_prob[i,j] = gray_prob[i,j]/(coarseness**2)
                white_prob[i,j] = white_prob[i,j]/(coarseness**2)
                skull_prob[i,j] = skull_prob[i,j]/(coarseness**2)
                csf_prob[i,j] = csf_prob[i,j]/(coarseness**2)
                scalp_prob[i,j] = scalp_prob[i,j]/(coarseness**2)

        gray_probs = np.zeros((cells_per_dim,cells_per_dim), dtype=int)
        labels = np.zeros((cells_per_dim,cells_per_dim), dtype=int)
        for i in range(cells_per_dim):
            for j in range(cells_per_dim):
                gray_probs[j,i] = gray_prob[i,j]
                if gray_prob[i,j]>0.5: 
                    labels[j,i]=1
                elif white_prob[i,j]>0.5: 
                    labels[j,i]=2
                elif skull_prob[i,j]>0.5: 
                    labels[j,i]=3
                elif csf_prob[i,j]>0.5: 
                    labels[j,i]=4
                elif scalp_prob[i,j]>0.5: 
                    labels[j,i]=5
                else:
                    labels[j,i]=0

        self.nodes = nodes_array.astype(int)
        self.elements = elements.astype(int)
        self.gray_probs = np.ravel(gray_probs, order='F')
        self.labels = np.ravel(labels, order='F').astype(int)

        np.savez_compressed(path, elements=self.elements, nodes=self.nodes, labels=self.labels, gray_probs=self.gray_probs)

        print("\nCreated new mesh with \n " + str(len(self.nodes)) + " nodes \n " + str(len(self.elements)) + " elements\n")
