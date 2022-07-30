#include "MUQ/Modeling/ModPiece.h"

#include <boost/property_tree/ptree.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

/***
## Class Definition

This class solves the 1D elliptic PDE of the form
    $$
    -\frac{\partial}{\partial x}\cdot(K(x) \frac{\partial h}{\partial x}) = f(x).
    $$
    over $x\in[0,1]$ with boundary conditions $h(0)=0$ and
    $\partial h/\partial x =0$ at $x=1$.  This equation is a basic model of steady
    state fluid flow in a porous media, where $h(x)$ is the hydraulic head, $K(x)$
    is the hydraulic conductivity, and $f(x)$ is the recharge.

    This ModPiece uses linear finite elements on a uniform grid. There is a single input,
    the conductivity $k(x)$, which is represented as piecewise constant within each
    of the $N$ cells.   There is a single output of this ModPiece: the head $h(x)$ at the
    $N+1$ nodes in the discretization.

*/
class FlowEquation : public muq::Modeling::ModPiece
{
public:

  /**
    INPUTS:
      @param[in] sourceTerm A numpy array containing the value of the source term $f(x)$ in each grid cell.
  */
  FlowEquation(Eigen::VectorXd const& sourceTerm) : muq::Modeling::ModPiece({int(sourceTerm.size())},
                                                                            {int(sourceTerm.size()+1)})
  {
    numCells = sourceTerm.size();
    numNodes = sourceTerm.size()+1;

    xs = Eigen::VectorXd::LinSpaced(numNodes,0,1);
    dx = xs(1)-xs(0);

    rhs = BuildRhs(sourceTerm);
  };

protected:

  /**
    Constructs the stiffness matrix and solves the resulting linear system.
    @param[in] inputs: A list of vector-valued model inputs.  Our model only has a single input,
                       so this list will also have one entry containing a vector of
                       conductivity values for each cell.
    @return This function returns nothing.  It stores the result in the private
            ModPiece::outputs list that is then returned by the `Evaluate` function.
  */
  void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) override
  {
    // Extract the conductivity vector from the inptus
    auto& cond = inputs.at(0).get();

    // Build the stiffness matrix
    auto K = BuildStiffness(cond);

    // Solve the sparse linear system and store the solution in the outputs vector
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K);

    outputs.resize(1);
    outputs.at(0) = solver.solve(rhs);
  };

  /**
    This function computes one step of the chain rule to compute the gradient
    $\nabla_k J$ of $J$ with respect to the input parameters.  In addition
    to the model parameter $k$, this function also requires the sensitivity of
    $J$ with respect to the flow equation output, i.e. $\nabla_h J$.

    The gradient with respect to the conductivity field is computed by solving
    the forward model, solving the adjoint system, and then combining the results to
    obtain the gradient.

    @param[in] outWrt For a model with multiple outputs, this would be the index
                      of the output list that corresponds to the sensitivity vector.
                      Since this ModPiece only has one output, the outWrt argument
                      is not used in the GradientImpl function.

    @param[in] inWrt Specifies the index of the input for which we want to compute
                     the gradient.  For inWrt==0, then the gradient with respect
                     to the conductivity is returned.  Since this ModPiece only has one
                     input, 0 is the only valid value for inWrt.

    @param[in] inputs Just like the EvalauteImpl function, this is a list of vector-valued inputs.

    @param[in] sensitivity A vector containing the gradient of an arbitrary objective $J$
                           with respect to the output of this ModPiece.

    @return This function returns nothing.  It stores the result in the private
        ModPiece::gradient variable that is then returned by the `Gradient` function.
  */
  virtual void GradientImpl(unsigned int outWrt,
                            unsigned int inWrt,
                            muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                            Eigen::VectorXd const& sens) override
  {
    // Extract the conductivity vector from the inptus
    auto& cond = inputs.at(0).get();

    // Build the stiffness matrix
    auto A = BuildStiffness(cond);

    // Factor the stiffness matrix for forward and adjoint solves
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    // Solve the forward problem
    Eigen::VectorXd sol = solver.solve(rhs);

    // Solve the adjoint problem
    Eigen::VectorXd adjRhs = BuildAdjointRHS(sens);
    Eigen::VectorXd adjSol = solver.solve(adjRhs);

    // Compute the gradient from the adjoint solution
    gradient = (1.0/dx)*(sol.tail(numCells)-sol.head(numCells)).array() * (adjSol.tail(numCells) - adjSol.head(numCells)).array();
  }


  /***
    This function computes the application of the model Jacobian's matrix $J$
    on a vector $v$.  In addition to the model parameter $k$, this function also
    requires the vector $v$.

    The gradient with respect to the conductivity field is computed by solving
    the forward model to get $h(x)$ and then using the tangent linear approach
    described above to obtain the Jacobian action.

    @param[in] outWrt For a model with multiple outputs, this would be the index
                      of the output list that corresponds to the sensitivity vector.
                      Since this ModPiece only has one output, the outWrt argument
                      is not used in the GradientImpl function.

    @param[in] inWrt Specifies the index of the input for which we want to compute
                     the gradient.  For inWrt==0, then the Jacobian with respect
                     to the conductivity is used.  Since this ModPiece only has one
                     input, 0 is the only valid value for inWrt.

    @param[in] inputs Just like the EvalauteImpl function, this is a list of vector-valued inputs.

    @param[in] vec A vector with the same size of inputs[0].  The Jacobian will be applied to this vector.

    @return This function returns nothing.  It stores the result in the private
            ModPiece::jacobianAction variable that is then returned by the `Jacobian` function.

  */
  virtual void ApplyJacobianImpl(unsigned int outWrt,
                                 unsigned int inWrt,
                                 muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                 Eigen::VectorXd const& vec) override
  {
    // Extract the conductivity vector from the inptus
    auto& cond = inputs.at(0).get();

    // Build the stiffness matrix
    auto A = BuildStiffness(cond);

    // Factor the stiffness matrix for forward and tangent linear solves
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    // Solve the forward system
    Eigen::VectorXd sol = solver.solve(rhs);

    // Build the tangent linear rhs
    Eigen::VectorXd incrRhs = Eigen::VectorXd::Zero(numNodes);

    Eigen::VectorXd dh_dx = (sol.tail(numCells)-sol.head(numCells))/dx; // Spatial derivative of solution

    incrRhs.head(numCells) += ( vec.array()*dh_dx.array() ).matrix();
    incrRhs.tail(numCells) -= ( vec.array()*dh_dx.array() ).matrix();

    // Solve the tangent linear model for the jacobian action
    jacobianAction = solver.solve(incrRhs);
  }

  /**
    Computes the action of Hessian on a vector.

    To understand this function, it is useful to interpret the `GradientImpl` function
    as a mapping $g(k,s)$ that accepts parameters $k$ as well as a sensitivity vector $s$
    and returns the gradient of $J$.  The Hessian matrix can be thought of as the Jacobian
    of this mapping $g(k,s)$.   This function therefore applies either the Jacobian with
    respect to $k$, $D_k g$, or the Jacobian with respect to $s$, $D_s g$,
    to a vector $v$.

    @param[in] outWrt For a model with multiple outputs, this would be the index
                       of the output list that corresponds to the sensitivity vector.
                       Since this ModPiece only has one output, the outWrt argument
                       is not used in the ApplyHessianImpl function.

    @param[in] inWrt1 Specifies the index of the first input that we want to differentiate
                      with respect to.  Since there is only one input to this ModPiece,
                      inWrt1 will also be 0.

    @param[in] inWrt2 Specifies the index of the second input that we want to differentiate
                      with respect to.  This second wrt argument can also be used to specify
                      derivatives with respect to the sensitivity input $s$.  For this example,
                      if `inWrt2=0`, then the Hessian $H_{kk}$ will be used.  If `inWrt2=1`, then
                      the second derivative will be with respect to $s$ and the Hessian $H_{ks}$
                      will be used.

    @param[in] inputs Just like the EvalauteImpl function, this is a list of vector-valued inputs.

    @param[in] sensitivity A vector containing the gradient of an arbitrary objective $J$
                           with respect to the output of this ModPiece.  This is the same as the
                           sensitivity vector used in the `GradientImpl` function.

    @param[in] vec The vector we want to apply the Hessian to.  The length of this vector should
                   be the same as the length of `inputs[0]` when inWrt2=0 and should be the same
                   length as `sensitivity` when `inWrt2=1`.

    @return This function returns nothing.  It stores the result in the `ApplyHessian` function.
  */
  virtual void ApplyHessianImpl(unsigned int outWrt,
                                 unsigned int inWrt1,
                                 unsigned int inWrt2,
                                 muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                 Eigen::VectorXd const& sens,
                                 Eigen::VectorXd const& vec) override
  {
    // Extract the conductivity vector from the inptus
    auto& cond = inputs.at(0).get();

    // Build the stiffness matrix
    auto K = BuildStiffness(cond);

    // Factor the stiffness matrix for forward and adjoint solves
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K);

    // Solve the forward problem
    Eigen::VectorXd sol = solver.solve(rhs);

    // If we're using the Hessian $\nabla_{kk} J$
    if((inWrt1==0)&&(inWrt2==0)){

      // Solve the adjoint problem
      Eigen::VectorXd adjRhs = BuildAdjointRHS(sens);
      Eigen::VectorXd adjSol = solver.solve(adjRhs);

      // Solve the incremental forward problem
      Eigen::VectorXd incrForRhs = BuildIncrementalRhs(sol, vec);
      Eigen::VectorXd incrForSol = solver.solve(incrForRhs);

      // Solve the incremental adjoint problem
      Eigen::VectorXd incrAdjRhs = BuildIncrementalRhs(adjSol, vec);
      Eigen::VectorXd incrAdjSol = solver.solve(incrAdjRhs);

      // Construct the Hessian action
      auto solDeriv = (sol.tail(numCells)-sol.head(numCells))/dx;
      auto adjDeriv = (adjSol.tail(numCells)-adjSol.head(numCells))/dx;
      auto incrForDeriv = (incrForSol.tail(numCells) - incrForSol.head(numCells))/dx;
      auto incrAdjDeriv = (incrAdjSol.tail(numCells) - incrAdjSol.head(numCells))/dx;

      hessAction = -(incrAdjDeriv.array() * solDeriv.array() + incrForDeriv.array() * adjDeriv.array());

    // If we're using the mixed Hessian $\nabla_{ks} J$
    }else if((inWrt1==0)&&(inWrt2==1)){

      Eigen::VectorXd temp = solver.solve(vec);
      auto solDeriv = (sol.tail(numCells) - sol.head(numCells))/dx;
      auto tempDeriv = (temp.tail(numCells)-temp.head(numCells))/dx;

      hessAction = -dx * solDeriv.array() * tempDeriv.array();

    // We should never see any other options...
    }else{
      assert(false);
    }
  }

  /** Construct the right hand side of the forward problem given a vector containing the source term f_i in each grid cell. */
  Eigen::VectorXd BuildRhs(Eigen::VectorXd const& sourceTerm) const
  {
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(numNodes);

    rhs.segment(1,numNodes-2) = 0.5*dx*(sourceTerm.tail(numNodes-2) + sourceTerm.head(numNodes-2));
    rhs(numNodes-1) = 0.5*dx*sourceTerm(numNodes-2);

    return rhs;
  }

  /** Construct the right hand side vector for the adjoint problem. */
  Eigen::VectorXd BuildAdjointRHS(Eigen::VectorXd const& sensitivity) const
  {
    Eigen::VectorXd rhs = -1.0*sensitivity;
    rhs(0) = 0.0; // <- To enforce Dirichlet BC
    return rhs;
  }

  /** Constructs the right hand side vector for both the incremental forward and incremental adjoint problems. */
  Eigen::VectorXd BuildIncrementalRhs(Eigen::VectorXd const& sol, Eigen::VectorXd const& khat)
  {
    // Compute the derivative of the solution in each cell
    Eigen::VectorXd solGrad = (sol.tail(numCells)-sol.head(numCells))/dx;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(numNodes);

    unsigned int leftNode, rightNode;
    for(unsigned int cellInd=0; cellInd<numCells; ++cellInd)
    {
      leftNode = cellInd;
      rightNode = cellInd + 1;

      rhs(leftNode) -= dx*khat(cellInd)*solGrad(cellInd);
      rhs(rightNode) += dx*khat(cellInd)*solGrad(cellInd);
    }

    rhs(0) = 0.0; // <- To enforce Dirichlet BC at x=0
    return rhs;
  }

  /** Build the sparse stiffness matrix given a vector of conductivities in each cell. */
  Eigen::SparseMatrix<double> BuildStiffness(Eigen::VectorXd const& condVals) const{

    typedef Eigen::Triplet<double> T;
    std::vector<T> nzVals;

    // Add a large number to K[0,0] to enforce Dirichlet BC
    nzVals.push_back( T(0,0,1e10) );

    unsigned int leftNode, rightNode;
    for(unsigned int cellInd=0; cellInd<numCells; ++cellInd)
    {
      leftNode  = cellInd;
      rightNode = cellInd+1;

      nzVals.push_back( T(leftNode,  rightNode, -condVals(cellInd)/dx) );
      nzVals.push_back( T(rightNode, leftNode,  -condVals(cellInd)/dx) );
      nzVals.push_back( T(rightNode, rightNode,  condVals(cellInd)/dx) );
      nzVals.push_back( T(leftNode,  leftNode,   condVals(cellInd)/dx) );
    }

    Eigen::SparseMatrix<double> stiffMat(numNodes,numNodes);
    stiffMat.setFromTriplets(nzVals.begin(), nzVals.end());

    return stiffMat;
  }


private:

  // Store "mesh" information.  xs contains the node locations and dx is the uniform spacing between nodes
  Eigen::VectorXd xs;
  double dx;
  unsigned int numCells;
  unsigned int numNodes;

  // Will store the precomputed RHS for the forward problem
  Eigen::VectorXd rhs;

}; // end of class SimpleModel
