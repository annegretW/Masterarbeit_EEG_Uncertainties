import numpy as np
from scipy import integrate

class StructuredMesh():
    def __init__(self, center, radii, cells_per_dim):
        self.center = center
        self.radii = radii
        self.cells_per_dim = cells_per_dim
        self.cell_size = 2*np.max(self.radii)/self.cells_per_dim

        N = cells_per_dim
        nr_nodes = N + 1
        lower_left = center - np.max(radii)
        upper_right = center + np.max(radii)

        # create nodes array for bounding box
        nodes = np.meshgrid(*[np.linspace(ll,ur,num=N+1) for ll,ur in zip(lower_left, upper_right)], indexing='ij')
        nodes_array = np.reshape(np.transpose(np.array(nodes, order='F')),[nr_nodes**3, 3], order = 'C')
        
        # map elements to nodes in bounding box
        elements = np.empty(shape=(N**3,8))
        idx_nodes = np.arange(nodes_array.shape[0])
        idx_back = idx_nodes[-nr_nodes**2:idx_nodes.size]
        idx_right = idx_nodes[idx_nodes%nr_nodes==nr_nodes-1]
        idx_top =  idx_nodes[idx_nodes%nr_nodes**2  >= (nr_nodes)*(nr_nodes-1) ]
        elements[:,0] = np.delete(idx_nodes,np.append(idx_back,np.append(idx_right, idx_top)))
        elements[:,1]     = elements[:,0] + 1
        elements[:,[2,3]] = elements[:,[0,1]] + nr_nodes
        elements[:,4:]    = elements[:,:4] + nr_nodes**2
        nr_elements = len(elements)

        # create tissue labels for elements in bounding box
        centers = np.meshgrid(*[ll+(ur-ll)*(np.linspace(1/N,1,N)-.5/N) for ll,ur in zip(lower_left, upper_right)], indexing = 'ij')
        dist = np.sqrt(sum([(x-c)**2 for x,c in zip(centers, center)]))
        labels = np.zeros([N]*len(lower_left), dtype=int)
        for i, r in enumerate(sorted(radii, reverse=True)):
            labels[dist<r] = i+1

        centers_array = np.reshape(np.transpose(np.array(centers, order='F')),[nr_elements, 3], order = 'C')
        labels_array= np.ravel(labels, order='F')

        # remove elements, nodes and labels outside sphere
        idx_elements_inside = np.nonzero(labels_array);
        labels_array = labels_array[labels_array != 0] - 1;
        elements_inside = np.squeeze(elements[idx_elements_inside,:])
        idx_nodes_inside = np.unique(np.ravel(elements_inside))
        palette = np.unique(np.ravel(elements_inside))
        key = np.arange(idx_nodes_inside.size)
        index = np.digitize(elements_inside.ravel(), palette, right=True)
        elements_inside = key[index].reshape(elements_inside.shape)

        #f = lambda y, x: np.sqrt(radii[0]-x**2-y**2)      
        #integrate.dblquad(f, self.centers[i][1]-(1/(2*N)), self.centers[i][1]+(1/(2*N)), self.centers[i][0]-(1/(2*N)), self.centers[i][0]+(1/(2*N)))

        self.nodes = nodes_array[idx_nodes_inside.astype(int),:]
        self.elements = elements_inside.astype(int)
        self.labels = np.ravel(labels_array, order='F')
        self.centers = centers_array[idx_elements_inside,:]

        print(self.centers)

        print("\nCreated new mesh with \n " + str(len(self.nodes)) + " nodes \n " + str(len(self.elements)) + " elements\n")

    def find_next_node(self, point):
        dist = np.sqrt(sum((self.centers[0][0]-point)**2))
        next = 0
        for i in range(len(self.centers[0])):
            if (np.sqrt(sum((self.centers[0][i]-point)**2)) < dist):
                dist = np.sqrt(sum((self.centers[0][i]-point)**2))
                next = i
        return next

    def find_next_center(self, point):
        p = np.array(point) - np.array(self.center) + np.max(self.radii)
        p = np.divmod(p, self.cell_size)[0]
        next = p*self.cell_size + self.cell_size/2 + np.array(self.center) - np.max(self.radii)
        return next