
#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/Modeling/UMBridge/UMBridgeModPiece.h"
#include "MUQ/Modeling/LinearAlgebra/HessianOperator.h"
#include "MUQ/Modeling/LinearAlgebra/StochasticEigenSolver.h"

#include <boost/property_tree/ptree.hpp>

/***
## Overview

The UM-Bridge interface allows coupling model and UQ codes through HTTP. A model may then
be implemented in virtually any programming language or framework, run in a container
or even on a remote machine. Likewise, the model does not make any assumptions on how the client is implemented.

This example shows how to connect to a running UM-Bridge server that is implemented in the UM-Bridge Server example.
The server provides the physical model, while the client is responsible for the UQ side.

The UM-Bridge interface is fully integrated in MUQ and can be used by means of the UMBridgeModPiece class.
Once such an UMBridgeModPiece is set up, it can be used like any other ModPiece. If the model supports the respective
functionality, the ModPiece then provides simple model evaluations,
gradient evaluations, applications of the Jacobian etc.

*/
int main(){

  using namespace muq::Modeling;

/***
## Connect to model server
First, we set up an UMBridgeModPiece that connects to our model server. This assumes that, before the client is started,
the model server from the UM-Bridge Server example is already running on your machine.
*/

  auto mod = std::make_shared<UMBridgeModPiece>("http://localhost:4242");

/***
## Set up dimensions
We then set up some helpers for determining dimensions and coordinates needed below.

Note that, from this point on, this example is completely identical to the FlowEquation example.
*/

  unsigned int numCells = 200;

  // Vector containing the location of each node in the "mesh"
  Eigen::VectorXd nodeLocs = Eigen::VectorXd::LinSpaced(numCells+1,0,1);

  // Vector containing the midpoint of each cell
  Eigen::VectorXd cellLocs = 0.5*(nodeLocs.head(numCells) + nodeLocs.tail(numCells));


/***
## Evaluate Model
Here we construct a vector of conductivity values and call the `FlowEquation.Evaluate` function to evaluate the model.
Recall that you should implement the `EvaluateImpl` function, but call the `Evaluate` function.
In addition to calling `EvaluateImpl`, the `Evaluate` function checks the size of the input vectors, tracks run
times, counts function evaluations, and manages a one-step cache of outputs.
*/

  // Define a conductivity field and evaluate the model k(x) = exp(cos(20*x))
  Eigen::VectorXd cond = (20.0*cellLocs.array()).cos().exp();

  Eigen::VectorXd h = mod->Evaluate(cond).at(0);

  std::cout << "Solution: \n" << h.transpose() << std::endl << std::endl;

/***
## Check Model Gradient
To check our adjoint gradient implementation, we will employ a finite difference approximation of the gradient vector.
Before doing that however, we need to define a scalar "objective function" $J(h)$ that can be composed with the flow
equation model.   In practice, this objective function is often the likelihood function or posterior density in a
Bayesian inverse problem.  For simplicity, we will just consider the log of a standard normal density:
$$
J(h) \propto -\frac{1}{2} \|h\|^2.
$$

The following cell uses the density view of MUQ's `Gaussian` class to define $J(h)$.   The `objective`
defined in this cell is just another `ModPiece` and be used in the same way as any other `ModPiece`.
*/
  auto objective = std::make_shared<Gaussian>(numCells+1)->AsDensity();

/***
To obtain the gradient of the objective function $J(h)$ with respect to the vector of cell-wise conductivities
$\mathbf{k}$, we need to apply the chain rule:
$$
\nabla_{\mathbf{k}} J = \left( \nabla_h J \right) \nabla_{\mathbf{k}} h
$$
The following cell uses the `objective` object to obtain the initial sensitivity $\nabla_h J$.  This is then
passed to the `Gradient` function of the flow model, which will use our adjoint implementation above to
compute $\nabla_{\mathbf{k}} J$.
*/
  Eigen::VectorXd objSens = Eigen::VectorXd::Ones(1);
  Eigen::VectorXd sens = objective->Gradient(0,0,h,objSens);
  Eigen::VectorXd grad = mod->Gradient(0,0,cond,sens);

  std::cout << "Gradient: \n" << grad.transpose() << std::endl << std::endl;

/***
To verify our `FlowEquation.GradientImpl` function, we can call the built in `ModPiece.GradientByFD` function
to construct a finite difference approximation of the gradient.  If all is well, the finite difference and adjoint
gradients will be close.
*/
  Eigen::VectorXd gradFD = mod->GradientByFD(0,0,std::vector<Eigen::VectorXd>{cond},sens);
  std::cout << "Finite Difference Gradient:\n" << gradFD.transpose() << std::endl << std::endl;

/***
## Test Jacobian of Model
Here we randomly choose a vector $v$ (`jacDir`) and compute the action of the Jacobian $Jv$ using both our adjoint method and MUQ's built-in finite difference implementation.
*/
  Eigen::VectorXd jacDir = Eigen::VectorXd::Ones(numCells);//RandomGenerator::GetUniform(numCells);

  Eigen::VectorXd jacAct = mod->ApplyJacobian(0,0, cond, jacDir);
  std::cout << "Jacobian Action: \n" << jacAct.transpose() << std::endl << std::endl;

  Eigen::VectorXd jacActFD = mod->ApplyJacobianByFD(0,0, std::vector<Eigen::VectorXd>{cond}, jacDir);
  std::cout << "Finite Difference Jacobian Action \n" << jacActFD.transpose() << std::endl << std::endl;

/***
## Test Hessian of Model
We now take a similar approach to verifying our Hessian action implementation.  Here we randomly choose a vector $v$ (`hessDir`)
and compute $Hv$ using both our adjoint method and MUQ's built-in finite difference implementation.
*/
  Eigen::VectorXd hessDir = Eigen::VectorXd::Ones(numCells);//RandomGenerator::GetUniform(numCells);

  Eigen::VectorXd hessAct = mod->ApplyHessian(0,0,0,cond,sens,hessDir);
  std::cout << "Hessian Action: \n" << hessAct.transpose() << std::endl << std::endl;

  Eigen::VectorXd hessActFD = mod->ApplyHessianByFD(0,0,0,std::vector<Eigen::VectorXd>{cond},sens,hessDir);
  std::cout << "Finite Difference Hessian Action \n" << hessActFD.transpose() << std::endl << std::endl;

/***
## Test Hessian of Objective
In the tests above, we manually evaluate the `objective` and `mod` components separately.   They can also be combined
in a MUQ `WorkGraph`, which is more convenient when a large number of components are used or Hessian information needs
to be propagated through multiple different components.

The following code creates a `WorkGraph` that maps the output of the flow model to the input of the objective function.
It then creates a new `ModPiece` called `fullMod` that evaluates the composition $J(h(k))$.
*/
  WorkGraph graph;
  graph.AddNode(mod, "Model");
  graph.AddNode(objective, "Objective");
  graph.AddEdge("Model",0,"Objective",0);

  auto fullMod = graph.CreateModPiece("Objective");

/***
As before, we can apply the Hessian of the full model to the randomly generated `hessDir` vector and compare the results
with finite differences.   Notice that the results shown here are slightly different than the Hessian actions computed above.
Above, we manually fixed the sensitivity $s$ independently of $h$ and did not account for the relationship between the
conductivity $k$ on the sensitivity $s$.   The `WorkGraph` however, captures all of those dependencies.
*/

  hessAct = fullMod->ApplyHessian(0,0,0,cond,objSens,hessDir);
  hessActFD = fullMod->ApplyHessianByFD(0,0,0,std::vector<Eigen::VectorXd>{cond},objSens,hessDir);

  std::cout << "Hessian Action: \n" << hessAct.transpose() << std::endl << std::endl;
  std::cout << "Finite Difference Hessian Action \n" << hessActFD.transpose() << std::endl << std::endl;


/***
## Compute the Hessian Spectrum
In many applications, the eigen decomposition of a Hessian matrix can contain valuable information.   For example,
let $\pi(y|k)$ denote the likelihood function in a Bayesian inverse problem.   The spectrum of $-\nabla_{kk}\log\pi(y|k)$
(the Hessian of the negative log likelihood) describes which directions in the parameter space are informed by the data.

Below we show how the spectrum of the Hessian can be computed with MUQ's stochastic eigenvalue solver.   MUQ's
implementation is based on the generalized eigenvalue solver described in
[Saibaba et al., 2015](https://doi.org/10.1002/nla.2026) and [Villa et al., 2021](https://dl.acm.org/doi/abs/10.1145/3428447?casa_token=2mk_QQqHe0kAAAAA%3AT5lr3QwgwbKNB4WgxUf7oPgCmqzir2b5O61fZHPEzv3AcU5eKHAxT1f7Ot_wZOL_SGqxe8g5ANAEtw).
*/

/***
#### Define the linear operator
The first step is to define a `LinearOperator` that evaluates the Hessian actions at a point.   Here, we create an
instance of MUQ's `HessianOperator` class, which will internally call the `ApplyHessian` function of the `fullMod` `ModPiece`.   The inputs to the `HessianOperator` constructor are essentially the same as the `ApplyHessian` function, but with an additional scaling term.   Here, `scaling=-1` is used to account for the fact that we want to use the Hessian of the *negative* log density, which will have a positive semi-definite Hessian.
*/
  unsigned int outWrt = 0; // There's only one output of "fullMod", so this is the only possibility
  unsigned int inWrt1 = 0; // There's also only one input of "fullMod"
  unsigned int inWrt2 = 0;

  double scaling = -1.0;
  std::vector<Eigen::VectorXd> inputs(1);
  inputs.at(0) = cond;
  auto hessOp = std::make_shared<HessianOperator>(fullMod, inputs, outWrt, inWrt1, inWrt2, objSens, scaling);

/***
#### Set up the eigenvalue solver
We can now set up the eigen solver, compute the decomposition, and extract the eigenvalues and eigenvectors.  For more
information, see documentation for the [StochasticEigenSolver](https://mituq.bitbucket.io/source/_site/latest/classmuq_1_1Modeling_1_1StochasticEigenSolver.html) class.
*/
  boost::property_tree::ptree opts;
  opts.put("NumEigs", numCells); // Maximum number of eigenvalues to compute
  opts.put("Verbosity", 3); // Controls how much information is printed to std::cout by the solver

  StochasticEigenSolver solver(opts);
  solver.compute(hessOp);

  Eigen::VectorXd vals = solver.eigenvalues();
  Eigen::MatrixXd vecs = solver.eigenvectors();

/***
#### Investigate the Spectrum
Below we plot the eigenvalues, which are sorted largest to smallest.  There are `numCells` parameters in this model
and `numCells+1` outputs.   The objective function in this example is similar to what we would see if all outputs
were observed.   The sharp decrease in the eigenvalues shown here is an indication that observing the `numCells+1`
outputs of this model is not sufficient to completely constrain all of the `numCells` inputs.  Without additional
regularization, an inverse problem using this model would therefore be ill-posed.  This is a common feature of
diffusive models.
*/
  std::cout << "Eigenvalues:\n" << vals.transpose() << std::endl << std::endl;
  return 0;
}
