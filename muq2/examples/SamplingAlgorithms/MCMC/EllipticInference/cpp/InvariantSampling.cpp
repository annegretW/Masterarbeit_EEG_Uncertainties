/***
## Background

A canonical inverse problem is to infer the spatially varying parameter field $K(x)$ in the following elliptic PDE
$$
-\nabla\cdot\left[ K \nabla h\right] = f,
$$
where $h(x)$ is the solution of the PDE and $f(x)$ is a source term.   We consider this PDE in a one dimensional (i.e., $x\in\mathbb{R}$) setting with boundary conditions given by
$$
\begin{aligned}
h(0) &= 0\\
\left. \frac{\partial h}{\partial x}\right|_{x=1} &= 0.
\end{aligned}
$$

This equation can be used to describe many different diffusive processes, such as heat conduction, groundwater flow, and contaminant diffusion.    This example interprets the PDE in the context of groundwater flow, meaning that $h(x)$ represents the hydraulic head in a confined aquifer, $f(x)$ represents recharge of water entering the aquifer, and $K(x)$ is the hydraulic conductivity, which describes how difficult it is for water to pass through the aquifer.

The hydraulic head $h(x)$ can be measured at individual points $x_1,x_2,\ldots,x_M$ by drilling boreholes into an aquifer and measuring the pressure in the borehole.  The conductivity $K(x)$ however, cannot be observed directly.   We will therefore consider estimating the hydraulic conductivity $K(x)$ from noisy measurements of $h(x_1), h(x_2),\ldots, h(x_M)$.



**The goal of this example is to demonstrate the use of MUQ for solving Bayesian inverse with PDE models.**   The implementation of the model itself will not be discussed in detail.   The implementation of the PDE solver, adjoint gradients, and Hessian information is described in detail in the [FlowEquation](../../../../Modeling/FlowEquation/python/DarcyFlow.ipynb) modeling example.

### Includes
This example will use many different components of MUQ as well as the `FlowEquation` class defined in [FlowEquation.h](FlowEquation.h). 
*/

#include "FlowEquation.h"

#include "MUQ/Approximation/GaussianProcesses/CovarianceKernels.h"
#include "MUQ/Approximation/GaussianProcesses/GaussianProcess.h"

#include "MUQ/Modeling/ModPiece.h"
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/LinearAlgebra/SliceOperator.h"
#include "MUQ/Modeling/Distributions/DensityProduct.h"
#include "MUQ/Modeling/CwiseOperators/CwiseUnaryOperator.h"

#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/DILIKernel.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/SampleCollection.h"
#include "MUQ/SamplingAlgorithms/Diagnostics.h"

#include "MUQ/Optimization/Optimizer.h"

#include "MUQ/Utilities/RandomGenerator.h"

#include <boost/property_tree/ptree.hpp>

using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Approximation;
using namespace muq::Utilities;
using namespace muq::Optimization;

/***
## Inverse Problem Formulation

For notational simplicity, let $\theta = \log K$.   In this example, we will define and subsequently sample a posterior density
$$
\pi(\theta | y=y_{obs}) \propto \pi(y=y_{obs} | \theta) \pi(\theta),
$$
where $y_{obs}$ is a vector of observed hydraulic heads at the points $x_1,\ldots, x_M$.   The observable variable $y$ is defined in terms of the PDE solution $h(x)$ evaluated at the observation locations:
$$
y = \left[ \begin{array}{c} h(x_1) + \epsilon_1\\ h(x_2)+ \epsilon_2\\ \vdots\\ h(x_M)+ \epsilon_M \end{array}\right],
$$
where each $\epsilon_i\sim N(0,\sigma_\epsilon^2)$ is an independent Gaussian random variable with variance $\sigma_\epsilon^2$.   This Gaussian assumption is common in practice, but may not be applicable in real-world situations where more complicated modeling errors and observation noise are present.   We will ignore these issues in this example.

For the prior, we will assume that $K(x)$ is a lognormal process and use MUQ's Gaussian process utilities to define $\pi(\theta)$.  

#### Discretization
We assume that $x\in\Omega=[0,1]$ and consider a uniform division of the domain $\Omega$ into $N$ sized cells.  There are $N+1$ nodes in the discretization.   To solve the PDE, we will use a linear finite element discretization.  The hydraulic head $h(x)$ is therefore piecewise linear and defined in terms of $N+1$ coefficients.  The hydraulic conductivity $K(x)$ is piecewise constant and can be represented with $N$ degrees of freedom. 

The `Discretization` class defined below holds information about the spatial discretization.
*/
struct Discretization
{   
    Discretization(unsigned int numCellsIn)
    {     
        numCells = numCellsIn;
        numNodes = numCells+1;

        nodeLocs = Eigen::VectorXd::LinSpaced(numCells+1,0,1);

        cellLocs.resize(1,numCells);
        cellLocs.row(0) = 0.5*(nodeLocs.tail(numCells) + nodeLocs.head(numCells));
    };

    unsigned int numCells;
    unsigned int numNodes;

    // Node locations 
    Eigen::VectorXd nodeLocs;

    // Cell locations in a row vector
    Eigen::MatrixXd cellLocs;
};


/***
#### Prior Distribution
By definition, the conductivity field $K(x)$ must be positive.   We will therefore define a prior distribution $\log K(x)$ to ensure that $K(x)>0$.   In particular, we will model the log conductivity field as a zero mean Gaussian process with covariance kernel $k(x,x^\prime)$:

$$
\log K(x) \sim GP(0,k(x,x^\prime))
$$

Here, we employ a Matern kernel with parameter $\nu=3/2$.  This kernel takes the form

$$
k(x,x^\prime) = \sigma^2 \left(1+\frac{\sqrt{3}\|x-x^\prime\|}{L}\right)\exp\left[-\frac{\sqrt{3}\|x-x^\prime\|}{L}\right],
$$

where $\sigma^2$ is the marginal variance of the prior distribution and $L$ is the correlation lengthscale of the kernel.  Note that this kernel results in random fields that have continuous first derivatives, but discontinuous second derivatives.   More generally, Matern kernels with $\nu=p+1/2$ result in fields with $p$ continuous derivatives.

The MUQ `GaussianProcess` class is constructed from a mean function and a covariance kernel.  The `GaussianProcess::Discretize` function can then to used to discretize the continuous Gaussian Process.  `Discretize` takes a vector of points and constructs a finite dimensional Gaussian distribution by evaluating the mean function at each point and the covariance kernel at each pair of points.  

Note that this is not always the best way to discretize a Gaussian Process.   For example, if the field $K(x)$ is represented with a finite number of basis functions $\phi_i(x)$, then it would be preferrable to project the Gaussian process onto the span of the basis functions.  This approach is skipped here for simplicity, but can be important, especially when $\phi_i(x)$ are high order or spectral basis functions.

<img src=Logk.svg width=600></img>

Our discretization of the PDE (see [DarcyFlow.ipynb](../../../../Modeling/FlowEquation/python/DarcyFlow.ipynb)) assumes that $K(x)$ is piecewise constant over grid cell (see Figure above).  Formally setting $\log K_i$ to the average of $\log K(x)$ over the cell $[x_i,x_{i+1})$ would result in a prior covariance matrix of the form

$$
\text{Cov}\left[\log K_i, \log K_j\right] = \frac{1}{(x_{i+1}-x_{i})(x_{j+1}-x_{j})}\int_{x_i}^{x_{i+1}} \int_{x_j}^{x_{j+1}} k(x_1,x_2) \,\,dx_2dx_1
$$

where $x_i$ is the location of the node on the left edge of cell $i$.  Here we approximate this covariance by simply evaluating the covariance kernel at the centroids of the cells: 

$$
\text{Cov}\left[\log K_i, \log K_j\right] \approx k\left(\frac{1}{2}(x_i + x_{i+1}),\,\, \frac{1}{2}(x_j + x_{j+1}) \right).
$$

Note that this is equivalent to approximating the integrals with a midpoint rule and a single interval.

**The vector $\log K = \left[\log K_1, \ldots, \log K_N\right]$ will be used to denote the collection of log conductivities on each cell.** 
*/
std::shared_ptr<Gaussian> CreatePrior(Discretization const& mesh)
{
    // Define the prior distribution
    double priorVar = 1.0;
    double priorLength = 0.2;
    double priorNu = 3.0/2.0;
    
    auto covKernel = std::make_shared<MaternKernel>(1, priorVar, priorLength, priorNu); // The first argument "1" specifies we are working in 1d
    
    auto meanFunc = std::make_shared<ZeroMean>(1,1); // dimension of x, components in k(x) if it was vector-valued

    auto priorGP = std::make_shared<GaussianProcess>(meanFunc,covKernel);

    return priorGP->Discretize(mesh.cellLocs);
}

/***
#### True data
After discretizing the PDE, the hydraulic head $h(x)$ is represented as a piecewise linear function characterized by values at the $N+1$ nodes in the discretization.    We will assume that every $P=\lceil (N+1)/M\rceil$ of these nodes is observed, resuling in a total of $M$ hydraulic noisy head observations.  Let $y_{obs}$ denote these observations.  

For this example, we will synthetically generate data $y_{obs}$ using a "true" log conductivity field $\log K(x) = \cos(10x)$ and a mesh with $2N$ cells.  Noise with variance $\sigma^2$ is added to the model output to simulate $y_{obs}$.
*/

Eigen::VectorXd GetTrueLogConductivity(Discretization const& mesh)
{
    return (10.0*mesh.cellLocs.row(0)).array().cos();
}

/***
The `SliceOperator` class in MUQ is used to downscale the model output.   Using numpy notation, the output of `SliceOperator` for an input vector $x$ is `x[startInd:endInd:skip]`.   The arguments to the `SliceOperator` constructor are `x.shape[0]`, `startInd`, `endInd`, and `skip`.
*/

Eigen::VectorXd GenerateData(Discretization const& mesh, unsigned int obsThin, double obsVar)
{   
    // Generate the data
    unsigned numRefine = 2;
    Discretization fineMesh(numRefine*mesh.numCells);
    Eigen::VectorXd trueCond = GetTrueLogConductivity(fineMesh).array().exp();

    // Create the model with twice the mesh resolution
    Eigen::VectorXd recharge = Eigen::VectorXd::Ones(fineMesh.numCells);
    auto mod = std::make_shared<FlowEquation>(recharge);

    // Solve the forward problem with the true conductivity
    Eigen::VectorXd trueSol = mod->Evaluate( trueCond ).at(0);

    // Take every N node as an "observation"
    auto slicer = std::make_shared<SliceOperator>(fineMesh.numNodes,0,fineMesh.numCells,numRefine*obsThin);

    return slicer->Evaluate(trueSol).at(0) + std::sqrt(obsVar)*RandomGenerator::GetNormal(slicer->outputSizes(0));
}

/***
#### Likelihood Function

To define a likelihood function $\pi(y|\theta)$, we need to construct a statistical model relating the PDE output $[h(x_0), h(x_P),\ldots, h(x_M)]$ to the observable quantity $y$.   Here, we assume that $y$ is  related to the hydraulic heads through an additive Gaussian error model

$$
y = \left[ \begin{array}{l} h(x_{0}; \theta)\\ h(x_{P}; \theta)\\ h(x_{2P}; \theta)\\ \vdots\\ h(x_M; \theta) \end{array} \right] + \epsilon,
$$

where $\epsilon\sim N(0,\sigma_{\epsilon}^2I)$ is an $M$-dimensional normal random variable with variance $\sigma_{\epsilon}^2$.   With this noise model, the distribution of the observable quantity $y$ given log conductivities $\theta$  is then just normal distribution centered at the model output

$$
\pi(y | \theta) = N\left([h(x_0), h(x_P),\ldots, h(x_M)], \,\, \sigma_{\epsilon}^2I\right)
$$

The `ConstructPosterior` function uses a MUQ `WorkGraph` to compose modeling components that together, compute the likelihood function, prior density, and posterior density.    In particular, operations for $\theta\rightarrow K$, $K\rightarrow h$, $h\rightarrow [h(x_0),\ldots, h(x_M)]$, and finally $[h(x_0), \ldots, h(x_M)]\rightarrow \pi(y | \theta)$.

The `DensityProduct` ModPiece is used for combining the prior and likelihood.  It takes 2 or more inputs representing log densities and returns the log of the density product (i.e., the sum of the log densities).

If [graphviz](https://graphviz.org/) is installed on your computer, the `graph.Visualize()` function will produce a file called "LikelihoodGraph.png" with a visualization of the components making up the likelihood function.
*/

std::shared_ptr<ModPiece> ConstructPosterior(Discretization             const& mesh,
                                             std::shared_ptr<Gaussian>  const& priorDist, 
                                             Eigen::VectorXd            const& data, 
                                             unsigned int                      obsThin, 
                                             double                            obsVar)
{   
    // Define the forward model
    Eigen::VectorXd recharge = Eigen::VectorXd::Ones(mesh.numCells);
    auto forwardMod = std::make_shared<FlowEquation>(recharge);
    
    auto graph = std::make_shared<WorkGraph>();

    // /////////////////////////////////////////////
    // LIKELIHOOD FUNCTION
    // Create a exponential operator for \theta -> K
    graph->AddNode(std::make_shared<IdentityOperator>(mesh.numCells), "Log Conductivity");
    graph->AddNode(std::make_shared<ExpOperator>(mesh.numCells), "Conductivity");
    graph->AddEdge("Log Conductivity", 0, "Conductivity", 0);

    // Add the forward model, which evaluates K->h
    graph->AddNode(forwardMod, "Forward Model");
    graph->AddEdge("Conductivity", 0, "Forward Model", 0);

    // Thin the model output to coincide with the observation locations 
    graph->AddNode(std::make_shared<SliceOperator>(mesh.numNodes,0,mesh.numCells,obsThin), "Observables");
    graph->AddEdge("Forward Model", 0, "Observables", 0);
    
    // Create the Gaussian likelihood that evaluates [h(x_0),\ldots, h(x_M)]\rightarrow \pi(y | \theta) 
    auto likelihood = std::make_shared<Gaussian>(data, obsVar*Eigen::VectorXd::Ones(data.size()));
    graph->AddNode(likelihood->AsDensity(), "Likelihood");
    graph->AddEdge("Observables", 0, "Likelihood", 0);

    // /////////////////////////////////////////////
    // PRIOR DENSITY
    graph->AddNode(priorDist->AsDensity(), "Prior");
    graph->AddEdge("Log Conductivity", 0, "Prior", 0);

    // /////////////////////////////////////////////
    // POSTERIOR DENSITY    
    graph->AddNode(std::make_shared<DensityProduct>(2), "Posterior");
    graph->AddEdge("Prior",0,"Posterior",0);
    graph->AddEdge("Likelihood",0,"Posterior",1);

    // Create a file LikelihoodGraph.png that visualizes the sequence of operations
    graph->Visualize("WorkGraph.pdf");

    return graph->CreateModPiece("Posterior");
}

/***
## Compute MAP Point
Now that we've define the posterior, we can start trying to estimate the parameters $\theta = \log K$ in the model.  The following cell computes the maximum aposterior (MAP) point using MUQ's Newton-Steihaug trust region optimizer.   The optimizer assumes it is given a minimization problem, while we want to maximize the log posterior density.  The `ModPieceCostFunction` class allows us to specify a negative scaling of the log posterior to flip the maximization problem into a minimization problem that the optimizer can handle.    

In addition to the trust region solver used here, MUQ also can also leverage any method implemented in [NLOPT](https://nlopt.readthedocs.io/en/latest/).  Try changing the `'NewtonTrust'` string to `'LBFGS'` to use the NLOPT limited memory BFGS implementation.
*/
Eigen::VectorXd ComputeMAP(std::shared_ptr<ModPiece> const& logPosterior, 
                           Eigen::VectorXd           const& startPt)
{
    std::cout << "\n======================================" << std::endl;
    std::cout << "Computing MAP Point" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    
    boost::property_tree::ptree opts;
    opts.put("Algorithm","NewtonTrust");
    opts.put("Ftol.AbsoluteTolerance", 1e-3);
    opts.put("PrintLevel", 1);
    
    // Create an objective function we can minimize -- the negative log posterior
    auto objective = std::make_shared<ModPieceCostFunction>(logPosterior, -1.0);
    auto solver = Optimizer::Construct(objective, opts);

    Eigen::VectorXd xopt;
    double fopt;
    std::tie(xopt,fopt) = solver->Solve({startPt});

    std::cout << "\nMAP Point: " << std::endl;
    std::cout << xopt.transpose() << std::endl;
    return xopt;
}

/***
## Sample Posterior with DILI

To sample the posterior distribution, we will employ MUQ's implementation of the dimension independent likelihood informed (DILI) method introduced in [Cui et al. 2015](https://arxiv.org/abs/1411.3688).   The idea is to decompose the parameter space into two subspaces: one "likelihood informed" subspace where the posterior distribution is significantly different than the prior, and a complementary subspace where the posterior distribution is approximately the same as the prior distribution.   Different MCMC kernels can be then be applied to each subspace.  If the MCMC sampler used in the complementary space is discretization invariant (e.g., the precondition Crank-Nicolson method), then the entire DILI sampling scheme is discretization invariant.

The subspaces are constructed by comparing the relative "strengths" of the likelihood and prior through the generalized eigenvalue problem 

$$
Hv = \lambda \Gamma^{-1}v,
$$

where $\Gamma^{-1}$ is the prior precision matrix, and $H$ is the Hessian of the negative log likelihood.  In this example, we will use the exact Hessian at the MAP point.  The `HessianType` option is therefore set to `Exact` in the `opt` dictionary below.   Gauss Newton approximations of the Hessian are also commonly employed in practice (see Eq. (19) of [Cui et al. 2015](https://arxiv.org/pdf/1411.3688.pdf)).  This can be accomplished in MUQ by changing the `HessianType` option to `GaussNewton`.    Note that the `Exact` Hessian is not always positive definite across the entire parameter space.   In this example, we are only using the Hessian at the MAP point, so this will not be an issue.  When the `Adapt Interval` option is nonzero however, the likelihood informed subspace will be formed from an average Hessian and, in that case, it is generally advisable to use the `GaussNewton` Hessian.


#### Choosing Starting Points
Metrics from MUQ's `MCMC::Diagnostics` class, like the multivariate potential scale reduction factor (MPSRF), can be used to evaluate whether our DILI sampler has converged.  To compute these metrics however, we need to run multiple chains from a diffuse set of initial points.  Generating these points in high dimensions can be challenging.   In this example, we use a Gaussian approximation of the posterior (i.e., the Laplace Approximation) constructed using the Hessian of the log-posterior at the MAP point.   Let $\theta^\ast$ denote the MAP point and $H$ the Hessian of the $2\pi(\theta | y=y_{obs})$.  The posterior can then be approximated as

$$
\pi(\theta | y=y_{obs}) \approx \tilde{\pi}(\theta|y=y_{obs}) = N(\theta^\ast, H^{-1})
$$

The DILI implementation in MUQ uses a low rank approximation of $H^{-1}$ to define the likelihood-informed and complementary subspaces.   We can use the same low rank structure to generate samples of the approximate posterior.  Let $\theta_{pr}\sim \pi(\theta)$ be a prior random variable and let $z_{lis}$ be a standard normal random variable in $\mathbb{R}^N_{LIS}$.  From these random variables we can construct a random variable $\theta_{app}\sim \tilde{\pi}(\theta|y=y_{obs})$ through

$$
\theta_{app} = \theta^\ast + Vz_{lis} + P(\theta_{pr}-\mu_{pr}),
$$

where the columns of $V$ span the likelihood-informed subspace while also accounting for posterior correlations and the matrix $P$ is an oblique projector onto the complementary space.   Internally, MUQ constructs the matrix $V$ by projecting $H^{-1}$ onto the likelihood-informed subspace and then using a matrix square root to decorrelate the projected random variable.

Fortunately, in this example we do not need consider the details of constructing $V$ and $P$.  Both of these are constructed in MUQ's `DILIKernel` class and can be applied using the `ToCS`, `ToLIS`, and `FromLIS` functions.  The `ToCS` function takes a vector in the full $N$ dimensional parameter space and projects it onto the prior-dominated complimentary space; it computes $P\theta_{pr}$  It returns another $N$ diemnsional vector.    The `ToLIS` function has a similar purpose: it takes and $N$ dimensional vector and projects it onto the likelihood-informed subspace.  However, `ToLIS` returns a vector with $N_{LIS}<N$ components.   The `FromLIS` function can then be used to map a point on the $N_{LIS}$-dimensional likelihood-informed subspace to the full $N$-dimensional space.

To generate diffuse initial starting points for the chains, we can "inflate" the variance of $z_{lis}$ and $\theta_{pr}$.   Below, the `SampleInflatedLaplace` function accomlishes this by multiplying $z_{lis}$ and $\theta_{pr}-\mu_{pr}$ by an inflation factor of $1.2$.
*/
Eigen::VectorXd SampleInflatedLaplace(std::shared_ptr<DILIKernel> const& diliKernel, 
                                      std::shared_ptr<Gaussian>   const& priorDist,
                                      Eigen::VectorXd             const& mapPt)
{
    
    double inflation = 1.2;
    
    diliKernel->CreateLIS(std::vector<Eigen::VectorXd>{mapPt});
    
    Eigen::VectorXd csSamp = diliKernel->ToCS( mapPt + inflation*(priorDist->Sample() - priorDist->GetMean()) );
    Eigen::VectorXd lisSamp = diliKernel->FromLIS( diliKernel->ToLIS(mapPt) + inflation*RandomGenerator::GetNormal(diliKernel->LISDim()) );
    
    return csSamp + lisSamp;
}

/***
#### Running the Sampler
We are now ready to define and run the DILI sampler.   The `SampleDili` function defines options for the MCMC samplers in both the likelihood-informed subspace and complementary space.  It then constructs the DILI sampler, samples the inflated Gaussian posterior approxiamtion to obtain an initial point, and runs the chain.   

Note that the pCN (`CrankNicolsonProposal`) used in the complementary space has a stepsize parameter of $\beta=1$.   This causes the pCN algorithm to use independent draws of the prior distribution as MCMC proposals.  Because the complementary space is approximately the same as prior distribution, this still results in large acceptance rates.
*/
std::shared_ptr<SampleCollection> SampleDILI(std::shared_ptr<ModPiece> const& posterior, 
                                             Eigen::VectorXd           const& mapPt,
                                             std::shared_ptr<Gaussian> const& priorDist,
                                             unsigned int                     numSamps)
{
    boost::property_tree::ptree pt;
    pt.put("NumSamples",numSamps);
    //pt.put("BurnIn", 0);
    pt.put("PrintLevel",3);
    pt.put("HessianType","Exact");
    pt.put("Adapt Interval", 0);
    pt.put("Initial Weight", 100);
    pt.put("Prior Node", "Prior");
    pt.put("Likelihood Node", "Likelihood");

    pt.put("LIS Block", "LIS");
    pt.put("LIS.Method", "MHKernel");
    pt.put("LIS.Proposal","MyProposal");
    pt.put("LIS.MyProposal.Method","MALAProposal");
    pt.put("LIS.MyProposal.StepSize", 0.4);

    pt.put("CS Block", "CS");
    pt.put("CS.Method", "MHKernel");
    pt.put("CS.Proposal","MyProposal");
    pt.put("CS.MyProposal.Method", "CrankNicolsonProposal");
    pt.put("CS.MyProposal.Beta",1.0);
    pt.put("CS.MyProposal.PriorNode","Prior");

    // create a sampling problem
    auto problem = std::make_shared<SamplingProblem>(posterior);

    auto diliKernel = std::make_shared<DILIKernel>(pt, problem);
    std::vector<std::shared_ptr<TransitionKernel>> kernels(1);
    kernels.at(0) = diliKernel;

    auto sampler = std::make_shared<SingleChainMCMC>(pt, kernels);

    Eigen::VectorXd startPt = SampleInflatedLaplace(diliKernel, priorDist, mapPt);

    return sampler->Run(startPt);
}

/***
## Sample Posterior with pCN
For comparison with the DILI results, we will use a standard preconditioned Crank-Nicolson proposal.   This method is discretization invariant, but does not leverage the same structure as DILI and has much poorer performance.   Other geometry-aware enhancements of the pCN proposal are also available in MUQ (e.g., [$\infty$-MALA](https://mituq.bitbucket.io/source/_site/latest/classmuq_1_1SamplingAlgorithms_1_1InfMALAProposal.html)), but are not shown in this example.

*/
std::shared_ptr<SampleCollection> SamplePCN(std::shared_ptr<ModPiece> const& posterior, 
                                            Eigen::VectorXd           const& startPt, 
                                            unsigned int                     numSamps)
{   
    boost::property_tree::ptree pt;
    pt.put("NumSamples", numSamps); // number of Monte Carlo samples
    pt.put("BurnIn",0);
    pt.put("PrintLevel",3);
    pt.put("KernelList", "Kernel1"); // the transition kernel
    pt.put("Kernel1.Method","MHKernel");
    pt.put("Kernel1.Proposal", "MyProposal"); // the proposal
    pt.put("Kernel1.MyProposal.Method", "CrankNicolsonProposal");
    pt.put("Kernel1.MyProposal.Beta", 0.05);
    pt.put("Kernel1.MyProposal.PriorNode", "Prior"); // The node in the WorkGraph containing the prior density

    // create a sampling problem
    auto problem = std::make_shared<SamplingProblem>(posterior);

    auto sampler = std::make_shared<SingleChainMCMC>(pt,problem);

    return sampler->Run(startPt); // Use a true posterior sample to avoid burnin
}


/***
## Put it all together
*/
int main(){

    
    // Define the mesh
    unsigned int numCells = 50;
    Discretization mesh(numCells);

    // Generate synthetic "truth" data
    unsigned int obsThin = 4;
    double obsVar = 0.01*0.01;
    auto data = GenerateData(mesh, obsThin, obsVar);
    
    // Create the priro distribution
    std::shared_ptr<Gaussian> prior = CreatePrior(mesh);

    // Construct the posterior
    std::shared_ptr<ModPiece> posterior = ConstructPosterior(mesh, prior, data, obsThin, obsVar);

    // Comptue the MAP point starting from the prior mean
    Eigen::VectorXd mapPt = ComputeMAP(posterior, prior->GetMean());

    // Run DILI multiple times
    unsigned int numSamps = 30000;
    unsigned int numChains = 4;
    std::vector<std::shared_ptr<SampleCollection>> chains(numChains);

    for(int i=0; i<numChains; ++i){
        std::cout << "\n===============================" << std::endl;
        std::cout << "Running DILI Chain " << i+1 << " / " << numChains << std::endl;
        std::cout << "-------------------------------" << std::endl;

        chains.at(i) = SampleDILI(posterior, mapPt, prior, numSamps);
    }
    
    /***
    #### Assess Convergence
    We now use the multiple DILI chains to assess convergence with the MPSRF diagnostic described in [[Brooks and Gelman, 1998](https://www.tandfonline.com/doi/abs/10.1080/10618600.1998.10474787)] and the Multivariate effective sample size of [[Vats et al., 2019](https://doi.org/10.1093/biomet/asz002)].   Note that when computing the MPSRF, we are also passing the `'Split':True` option, which follows the suggestions of [[Vehtari et al., 2021](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1214%2F20-BA1221&v=8607c76e)] and splits each chain in half.
    */
    double diliESS = 0;
    for(auto& chain : chains)
        diliESS += chain->ESS("MultiBatch")(0);

    boost::property_tree::ptree opts;
    opts.put("Split", true);
    opts.put("Transform", false);
    opts.put("Multivariate",true);

    Eigen::VectorXd diliMPSRF = Diagnostics::Rhat(chains, opts);

    std::cout << "\n\nDILI Diagnostics:\n  Multivariate ESS: " << diliESS << "\n  MPSRF: " << diliMPSRF(0) << std::endl;


    if(diliMPSRF(0)>1.01){
        std::cout << "\nDILI HAS NOT CONVERGED!" << std::endl;
    }else{
        std::cout << "\nDILI CONVERGED!" << std::endl;
    }


    /***
    #### Run pCN for comparison
    */
    std::vector<std::shared_ptr<SampleCollection>> pcnChains(numChains);

    for(int i=0; i<numChains; ++i){
        std::cout << "\n===============================" << std::endl;
        std::cout << "Running pCN Chain " << i+1 << " / " << numChains << std::endl;
        std::cout << "-------------------------------" << std::endl;
    
        pcnChains.at(i) = SamplePCN(posterior, chains.at(i)->at(0)->ToVector(), numSamps);
    }


    double pcnESS = 0;
    for(auto& chain : pcnChains)
        pcnESS += chain->ESS("MultiBatch")(0);

    Eigen::VectorXd pcnMPSRF = Diagnostics::Rhat(pcnChains, opts);

    std::cout << "\n\npCN Diagnostics:\n  Multivariate ESS: " << pcnESS << "\n  MPSRF: " << pcnMPSRF(0) << std::endl;


    if(pcnMPSRF(0)>1.01){
        std::cout << "\npCN HAS NOT CONVERGED!" << std::endl;
    }else{
        std::cout << "\npCN CONVERGED!" << std::endl;
    }
    return 0;
}