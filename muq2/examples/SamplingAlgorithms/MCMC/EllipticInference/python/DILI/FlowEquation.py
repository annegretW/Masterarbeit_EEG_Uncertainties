import muq.Modeling as mm

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spla



class FlowEquation(mm.PyModPiece):
    """
    This class solves the 1D elliptic PDE of the form
    $$
    -\frac{\partial}{\partial x}\cdot(k(x) \frac{\partial h}{\partial x}) = f(x).
    $$
    over $x\in[0,1]$ with boundary conditions $h(0)=0$ and
    $\partial h/\partial x =0$ at $x=1$.  This equation is a basic model of steady
    state fluid flow in a porous media, where $h(x)$ is the hydraulic head, $K(x)$
    is the hydraulic conductivity, and $f(x)$ is the recharge.

    This ModPiece uses linear finite elements on a uniform grid. There is a single input,
    the conductivity $K(x)$, which is represented as piecewise constant within each
    of the $N$ cells.   There is a single output of this ModPiece: the head $h(x)$ at the
    $N+1$ nodes in the discretization.
    
    INPUTS:
        sourceTerm (np.array) : A numpy array containing the value of the source term $f(x)$ in each grid cell.

    """

    def __init__(self, sourceTerm):
        super(FlowEquation,self).__init__([sourceTerm.shape[0]],   # inputSizes  (one for each cell)
                                          [sourceTerm.shape[0]+1]) # outputSizes (one for each node)
        
        self.numCells = sourceTerm.shape[0]
        self.numNodes = self.numCells+1
        
        self.xs = np.linspace(0,1,self.numNodes) # Assumes domain is [0,1]
        self.dx = self.xs[1]-self.xs[0] 
        
        self.rhs = self.BuildRhs(sourceTerm)
        
        
    def EvaluateImpl(self, inputs):
        """ Constructs the stiffness matrix and solves the resulting linear system.

            INPUTS:
                inputs: A list of vector-valued inputs.  Our model only has a single input,
                        so this list will also have one entry containing a vector of
                        conductivity values for each cell.

            RETURNS:
                This function returns nothing.  It stores the result in the private
                ModPiece::outputs list that is then returned by the `Evaluate` function.
        """

        condVals = inputs[0]

        # Build the stiffness matrix and right hand side
        A = self.BuildStiffness(condVals)

        # Solve the sparse linear system
        sol = spla.spsolve(A,self.rhs)

        # Set the output list using the solution
        self.outputs = [sol]

        
    def GradientImpl(self, outWrt, inWrt, inputs, sensitivity):
        """ This function computes one step of the chain rule to compute the gradient 
            $\nabla_k J$ of $J$ with respect to the input parameters.  In addition 
            to the model parameter $k$, this function also requires the sensitivity of 
            $J$ with respect to the flow equation output, i.e. $\nabla_h J$.
        
            The gradient with respect to the conductivity field is computed by solving
            the forward model, solving the adjoint system, and then combining the results to
            obtain the gradient.

            INPUTS:
                outWrt: For a model with multiple outputs, this would be the index
                        of the output list that corresponds to the sensitivity vector.
                        Since this ModPiece only has one output, the outWrt argument
                        is not used in the GradientImpl function.

                inWrt: Specifies the index of the input for which we want to compute
                       the gradient.  For inWrt==0, then the gradient with respect
                       to the conductivity is returned.  Since this ModPiece only has one 
                       input, 0 is the only valid value for inWrt.

                inputs: Just like the EvalauteImpl function, this is a list of vector-valued inputs.

                sensitivity: A vector containing the gradient of an arbitrary objective $J$
                             with respect to the output of this ModPiece.

            RETURNS:
                This function returns nothing.  It stores the result in the private
                ModPiece::gradient variable that is then returned by the `Gradient` function.

        """
        condVals = inputs[0]

        # Construct the adjoint system
        A = self.BuildStiffness(condVals)
        adjRhs = self.BuildAdjointRhs(sensitivity)
        
        # Solve the adjoint system
        sol = spla.spsolve(A,self.rhs)
        adjSol = spla.spsolve(A,adjRhs)

        # Compute the gradient from the adjoint solution
        dhdx = (sol[1:] - sol[:-1])/self.dx   # derivative of forward solution, which is constant in each cell
        dpdx = (adjSol[1:] - adjSol[:-1])/self.dx # derivative of adjoint solution, which is constant in each cell
        
        self.gradient = self.dx * dhdx * dpdx 
        
    def ApplyJacobianImpl(self, outWrt, inWrt, inputs, vec):
        """ This function computes the application of the model Jacobian's matrix $J$
            on a vector $v$.  In addition to the model parameter $k$, this function also
            requires the vector $v$. 
        
            The gradient with respect to the conductivity field is computed by solving
            the forward model to get $h(x)$ and then using the incremental approach 
            described above to obtain the Jacobian action.

            INPUTS:
                outWrt: For a model with multiple outputs, this would be the index
                        of the output list that corresponds to the sensitivity vector.
                        Since this ModPiece only has one output, the outWrt argument
                        is not used in the GradientImpl function.

                inWrt: Specifies the index of the input for which we want to compute
                       the gradient.  For inWrt==0, then the Jacobian with respect
                       to the conductivity is used.  Since this ModPiece only has one 
                       input, 0 is the only valid value for inWrt.

                inputs: Just like the EvalauteImpl function, this is a list of vector-valued inputs.

                vec: A vector with the same size of inputs[0].  The Jacobian will be applied to this vector.
                
            RETURNS:
                This function returns nothing.  It stores the result in the private
                ModPiece::jacobianAction variable that is then returned by the `Jacobian` function.

        """
        
        condVals = inputs[0]
        
        # Build the stiffness matrix
        A = self.BuildStiffness(condVals)
        
        # Solve the forward system
        sol = spla.spsolve(A,self.rhs)
        
        # Build the rhs 
        incrRhs = np.zeros(self.numNodes)
        
        dh_dx = (sol[1:]-sol[:-1])/self.dx  # Spatial derivative of solution
        
        incrRhs[:-1] += vec*dh_dx
        incrRhs[1:] -= vec*dh_dx
            
        # Solve the incremental problem for the jacobian action
        self.jacobianAction = spla.spsolve(A, incrRhs)
        
    def ApplyHessianImpl(self, outWrt, inWrt1, inWrt2, inputs, sensitivity, vec):
        """  Computes the action of Hessian on a vector.
        
             To understand this function, it is useful to interpret the `GradientImpl` function 
             as a mapping $g(k,s)$ that accepts parameters $k$ as well as a sensitivity vector $s$ 
             and returns the gradient of $J$.  The Hessian matrix can be thought of as the Jacobian 
             of this mapping $g(k,s)$.   This function therefore applies either the Jacobian with 
             respect to $k$, $D_k g$, or the Jacobian with respect to $s$, $D_s g$, 
             to a vector $v$.   

            INPUTS:
                outWrt: For a model with multiple outputs, this would be the index
                        of the output list that corresponds to the sensitivity vector.
                        Since this ModPiece only has one output, the outWrt argument
                        is not used in the ApplyHessianImpl function.

                inWrt1: Specifies the index of the first input that we want to differentiate 
                        with respect to.  Since there is only one input to this ModPiece,
                        inWrt1 will also be 0.
                
                inWrt2: Specifies the index of the second input that we want to differentiate 
                        with respect to.  This second wrt argument can also be used to specify 
                        derivatives with respect to the sensitivity input $s$.  For this example,
                        if `inWrt2=0`, then the Hessian $H_{kk}$ will be used.  If `inWrt2=1`, then
                        the second derivative will be with respect to $s$ and the Hessian $H_{ks}$ 
                        will be used.

                inputs: Just like the EvalauteImpl function, this is a list of vector-valued inputs.

                sensitivity: A vector containing the gradient of an arbitrary objective $J$
                             with respect to the output of this ModPiece.  This is the same as the 
                             sensitivity vector used in the `GradientImpl` function.
                             
                vec : The vector we want to apply the Hessian to.  The length of this vector should 
                      be the same as the length of `inputs[0]` when inWrt2=0 and should be the same 
                      length as `sensitivity` when `inWrt2=1`.

            RETURNS:
                This function returns nothing.  It stores the result in the `ApplyHessian` function.

        """
        
        condVals = inputs[0]
        
        # Build the stiffness matrix
        A = self.BuildStiffness(condVals)
        
        # Solve the forward system
        sol = spla.spsolve(A,self.rhs)

        # If we're using the Hessian $\nabla_{kk} J$
        if((inWrt1==0)&(inWrt2==0)):
                        
            # Solve the adjoint system
            adjRhs = self.BuildAdjointRhs(sensitivity)
            adjSol = spla.spsolve(A,adjRhs) # Because A is symmetric

            # Solve the incremental forward system
            incrForRhs = self.BuildIncrementalRhs(sol, vec)
            incrForSol = spla.spsolve(A,incrForRhs)
            
            # Solve the incremental adjoint system
            incrAdjRhs = self.BuildIncrementalRhs(adjSol,vec)
            incrAdjSol = spla.spsolve(A,incrAdjRhs) # Because A is symmetric
            
            # Construct the Hessian action
            solDeriv = (sol[1:]-sol[:-1])/self.dx
            adjDeriv = (adjSol[1:]-adjSol[:-1])/self.dx
            incrForDeriv = (incrForSol[1:]-incrForSol[:-1])/self.dx
            incrAdjDeriv = (incrAdjSol[1:]-incrAdjSol[:-1])/self.dx
            
            self.hessAction = (incrAdjDeriv * solDeriv + incrForDeriv * adjDeriv)#*2.
            
        # If we're using the mixed Hessian $\nabla_{ks} J$
        elif(((inWrt1==0)&(inWrt2==1))|((inWrt1==1)&(inWrt2==0))):
            
            temp = spla.spsolve(A, vec)
            dhdx = (sol[1:] - sol[:-1])/self.dx   # derivative of forward solution, which is constant in each cell
            dtempdx = (temp[1:] - temp[:-1])/self.dx # derivative of adjoint solution, which is constant in each cell
            
            self.hessAction = -self.dx * dhdx * dtempdx 
        
        else: 
            assert(False)

            
    def BuildStiffness(self, condVals):
        """ Constructs the stiffness matrix for the conductivity defined within each cell. """

        rows = []
        cols = []
        vals = []
        
        # Left Dirichlet BC
        rows.append(0)
        cols.append(0)
        vals.append(1e10) # Trick to approximately enforce Dirichlet BC at x=0 while keeping matrix symmetric

        # Integration over each cell
        for cellInd in range(self.numCells):
            leftNode = cellInd
            rightNode = cellInd+1
            
            rows.append(leftNode)
            cols.append(rightNode)
            vals.append(-condVals[cellInd]/self.dx)
            
            rows.append(rightNode)
            cols.append(leftNode)
            vals.append(-condVals[cellInd]/self.dx)
            
            rows.append(rightNode)
            cols.append(rightNode)
            vals.append(condVals[cellInd]/self.dx)
            
            rows.append(leftNode)
            cols.append(leftNode)
            vals.append(condVals[cellInd]/self.dx)

        return sp.csr_matrix((vals,(rows,cols)), shape=(self.numNodes, self.numNodes))
    
    
    def BuildRhs(self, recharge):
        """ Constructs the right hand side vector for the forward problem. """

        rhs = np.zeros((self.numNodes,))

        rhs[1:-1] = 0.5*self.dx*(recharge[0:-1] + recharge[1:])
        rhs[-1] = 0.5*self.dx*recharge[-1]
        return rhs

    
    def BuildAdjointRhs(self,sensitivity):
        """ Constructs the right hand side vector for the adjoint problem. """
        rhs = -1.0*sensitivity 
        rhs[0] = 0.0
        return rhs

    
    def BuildIncrementalRhs(self, sol, khat):
        """ Constructs the right hand side vector for both incremental problems. """
        # The derivative of the solution in each cell
        solGrad = (sol[1:]-sol[:-1])/self.dx
        rhs = np.zeros((self.numNodes,))
        
        for cellInd in range(self.numCells):
            leftNode = cellInd
            rightNode = cellInd+1
            
            rhs[leftNode] -= self.dx*khat[cellInd]*solGrad[cellInd]
            rhs[rightNode] += self.dx*khat[cellInd]*solGrad[cellInd]
            
        rhs[0]=0.0
        return -rhs
    