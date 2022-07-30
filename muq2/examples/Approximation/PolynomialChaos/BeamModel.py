import muq.Modeling as mm
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class BeamModel(mm.PyModPiece):
    """
    This class uses a second order finite difference approximate the displacement
    of an Euler-Bernoulli with piecewise constant stiffness.

    ## Finite Difference Discretization
    For the interior nodes, second order finite difference approximations to the
    derivatives yields:
    $$
    \frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1}-2u_i+u_{i-1}}{\delta x^2}
    $$
    which then implies
    $$
    \frac{\partial^2}{\partial x^2}\left[k(x) \frac{\partial^2 u}{\partial x^2}\right] \approx \frac{1}{\delta x^2}\left[k_{i+1}\frac{u_{i+2}-2u_{i+1}+u_{i}}{\delta x^2} - 2k_{i}\frac{u_{i+1}-2u_{i}+u_{i-1}}{\delta x^2} + k_{i-1}\frac{u_{i}-2u_{i-1}+u_{i-2}}{\delta x^2}\right] = \frac{1}{\delta x^4}\left[k_{i+1}u_{i+2} - 2(k_{i+1}+k_i)u_{i+1} + (k_{i+1}+4k_i+k_{i-1})u_{i} - 2(k_i+k_{i-1}) u_{i-1} + k_{i-1} u_{i-2}\right]
    $$.

    One node in from the left boundary, we have the boundary condition
    $$
    \frac{\partial u}{\partial x} = 0,
    $$
    which we need to use in our finite difference discretization.  Consider the
    one sided second order approximation
    $$
    \frac{\partial^2}{\partial x^2}\left[k(x) \frac{\partial^2 u}{\partial x^2}\right] \approx \frac{1}{\delta x^2}\left[2k(x_1)\left.\frac{\partial^2 u}{\partial x^2}\right|_{x_1} - 5k(x_2)\left.\frac{\partial^2 u}{\partial x^2}\right|_{x_2} + 4k(x_3)\left.\frac{\partial^2 u}{\partial x^2}\right|_{x_3}-k(x_4)\left.\frac{\partial^2 u}{\partial x^2}\right|_{x_4}\right]
    $$
    The boundary condition implies
    $$
    \left.\frac{\partial^2 u}{\partial x^2}\right|_{x_1} \approx \frac{1}{2\delta x}\left[\left.\frac{\partial u}{\partial x}\right|_{x_2} - \left.\frac{\partial u}{\partial x}\right|_{x_0}\right] \approx \frac{u_3-u_1}{4\delta x^2},
    $$
    which can be combined with standard finite difference estimates of the second
    derivatives to obtain
    $$
    \frac{\partial^2}{\partial x^2}\left[k(x) \frac{\partial^2 u}{\partial x^2}\right] \approx \frac{1}{\delta x^4}\left[\frac{1}{2}k_1(u_3-u_1) - 5k_2(u_3-2u_2+u_1) + 4k_3(u_2-2u_3+u_4)-k_4(u_5-2u_4+u_3)\right] = \frac{1}{\delta x^4}\left[(-\frac{1}{2}k_1-5k_2)u_1 + (10k_2+4k_3)u_2 + (\frac{1}{2}k_1-5k_2-8k_3-k_4)u_3 + (4k_3+2k_4)u_4 -k_4u_5\right]
    $$
    """

    def __init__(self, x, numBlocks, radius=0.1):
        mm.PyModPiece.__init__(self, [1,numBlocks], [x.shape[0]])# One output (the displacement)
        self.x = x

        # Moment of inertia of beam (assuming cylindrical beam)
        self.moi = np.pi/4.0*radius**4

        self.numBlocks = numBlocks
        self.numNodes = x.shape[0]
        nodesPerBlock = int(self.numNodes / numBlocks)
        self.blockStarts = [i*nodesPerBlock for i in range(numBlocks)] + [self.numNodes+1]
        print(self.blockStarts)

    def EvaluateImpl(self, inputs):

        # Distributed load over the beam (np.array)
        load = inputs[0][0]*np.ones(self.numNodes)/self.moi
        load[0] = 0.0 # Needed to enforce the Dirichlet BC on the left
        load[1] = 0.0 # Needed to enforce the zero derivative BC on the left

        # Construct a vector of stiffnesses at every node
        logStiffness = inputs[1]
        stiffVec = np.zeros(self.numNodes,)
        for i in range(self.numBlocks):
            stiffVec[self.blockStarts[i]:self.blockStarts[i+1]] = np.exp(logStiffness[i])

        # Construct the stiffness matrix
        K = self.BuildStiffness(stiffVec)

        # Solve system to get displacement
        displacement = spla.spsolve(K, load,use_umfpack=True)
        self.outputs = [displacement]

    def BuildStiffness(self,modulus):
        """ Construct the sparse stiffness matrix given a vector containing exp(m)
            at every finite difference node
        """
        dx = self.x[1]-self.x[0]
        numPts = self.x.shape[0]

        dx4 = dx**4

        # Create stiffness matrix
        rows = []
        cols = []
        vals = []
        #K = np.zeros((numPts, numPts))

        # Build stiffness matrix (center)
        for i in range(2, numPts-2):
            rows.append(i)
            cols.append(i+2)
            vals.append(modulus[i+1] / dx4)

            rows.append(i)
            cols.append(i+1)
            vals.append(-2.0*(modulus[i+1] + modulus[i]) / dx4)

            rows.append(i)
            cols.append(i)
            vals.append((modulus[i+1] + 4.0*modulus[i] + modulus[i-1]) / dx4)

            rows.append(i)
            cols.append(i-1)
            vals.append(-2.0*(modulus[i] + modulus[i-1]) / dx4)

            rows.append(i)
            cols.append(i-2)
            vals.append(modulus[i-1] / dx4)


        # Set row i == 1
        rows.append(1)
        cols.append(1)
        vals.append((-0.5*modulus[1]-5*modulus[2]) / dx4)

        rows.append(1)
        cols.append(2)
        vals.append((10.0*modulus[2] + 4.0*modulus[3])/dx4)

        rows.append(1)
        cols.append(3)
        vals.append((0.5*modulus[1]-5.0*modulus[2]-8.0*modulus[3]-modulus[4])/dx4)

        rows.append(1)
        cols.append(4)
        vals.append((4.0*modulus[3] + 2.0*modulus[4])/dx4)

        rows.append(1)
        cols.append(5)
        vals.append(-modulus[4]/dx4)

        rows.append(numPts-2)
        cols.append(numPts-1)
        vals.append((modulus[numPts-1] - 4.0*modulus[numPts-2] + modulus[numPts-3])/dx4)

        rows.append(numPts-2)
        cols.append(numPts-2)
        vals.append((-2.0*modulus[numPts-1] + 9.0*modulus[numPts-2] - 2.0*modulus[numPts-3])/dx4)

        rows.append(numPts-2)
        cols.append(numPts-3)
        vals.append((modulus[numPts-1] - 6.0*modulus[numPts-2] + modulus[numPts-3])/dx4)

        rows.append(numPts-2)
        cols.append(numPts-4)
        vals.append(modulus[numPts-2]/dx4)

        rows.append(numPts-1)
        cols.append(numPts-1)
        vals.append(2.0*modulus[numPts-1]/dx4)

        rows.append(numPts-1)
        cols.append(numPts-2)
        vals.append(-4.0*modulus[numPts-1] / dx4)

        rows.append(numPts-1)
        cols.append(numPts-3)
        vals.append(2.0*modulus[numPts-1]/dx4)

        # Apply dirichlet BC (w=0 at x=0)
        rows.append(0)
        cols.append(0)
        vals.append(1)

        return sp.csr_matrix((vals,(rows,cols)), shape=(numPts,numPts))
