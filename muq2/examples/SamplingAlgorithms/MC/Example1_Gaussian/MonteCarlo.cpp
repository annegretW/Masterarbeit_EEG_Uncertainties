#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"

#include "MUQ/SamplingAlgorithms/DummyKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/MCMCFactory.h"

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

#include "MCSampleProposal.h" // TODO: Move into muq

class MySamplingProblem : public AbstractSamplingProblem {
public:
  MySamplingProblem()
   : AbstractSamplingProblem(Eigen::VectorXi::Constant(1,1), Eigen::VectorXi::Constant(1,1))
     {}

  virtual ~MySamplingProblem() = default;


  virtual double LogDensity(std::shared_ptr<SamplingState> const& state) override {
    lastState = state;
    return 0;
  };

  virtual std::shared_ptr<SamplingState> QOI() override {
    assert (lastState != nullptr);
    return std::make_shared<SamplingState>(lastState->state[0] * 2, 1.0);
  }

private:
  std::shared_ptr<SamplingState> lastState = nullptr;

};


int main(){

  /***
  ### 1. Define the target density and set up sampling problem
  MUQ has extensive tools for combining many model compoenents into larger
  more complicated models.  The AbstractSamplingProblem base class and its
  children, like the SamplingProblem class, define the interface between
  sampling algorithms like MCMC and the models and densities they work with.

  Here, we create a very simple target density and then construct a SamplingProblem
  directly from the density.
  */
  /***
  Define the Target Density:
  */
  Eigen::VectorXd mu(2);
  mu << 1.0, 2.0;

  Eigen::MatrixXd cov(2,2);
  cov << 1.0, 0.8,
         0.8, 1.5;

  auto targetDensity = std::make_shared<Gaussian>(mu, cov)->AsDensity(); // standard normal Gaussian

  /***
  Create the Sampling Problem:
  */
  //auto problem = std::make_shared<SamplingProblem>(targetDensity, targetDensity);

  auto problem = std::make_shared<MySamplingProblem>();

  /***
  ### 2. Construct the RWM algorithm
  One of the easiest ways to define an MCMC algorithm is to put all of the algorithm
  parameters, including the kernel and proposal definitions, in a property tree
  and then let MUQ construct each of the algorithm components: chain, kernel, and
  proposal.

  The boost property tree will have the following entries:

  - NumSamples : 10000
  - KernelList "Kernel1"
  - Kernel1
    * Method : "MHKernel"
    * Proposal : "MyProposal"
    * MyProposal
      + Method : "MHProposal"
      + ProposalVariance : 0.5

At the base level, we specify the number of steps in the chain with the entry "NumSamples".
  Note that this number includes any burnin samples.   The kernel is then defined
  in the "KernelList" entry.  The value, "Kernel1", specifies a block in the
  property tree with the kernel definition.  In the "Kernel1" block, we set the
  kernel to "MHKernel," which specifies that we want to use the Metropolis-Hastings
  kernel.  We also tell MUQ to use the "MyProposal" block to define the proposal.
  The proposal method is specified as "MHProposal", which is the random walk
  proposal used in the RWM algorithm, and the proposal variance is set to 0.5.
  */

  // parameters for the sampler
  pt::ptree pt;
  pt.put("NumSamples", 1e4); // number of MCMC steps
  pt.put("BurnIn", 1e3);
  pt.put("PrintLevel",3);
  /*pt.put("KernelList", "Kernel1"); // Name of block that defines the transition kernel
  pt.put("Kernel1.Method","MHKernel");  // Name of the transition kernel class
  pt.put("Kernel1.Proposal", "MyProposal"); // Name of block defining the proposal distribution
  pt.put("Kernel1.MyProposal.Method", "MHProposal");*/ // Name of proposal class

  /***
  Once the algorithm parameters are specified, we can pass them to the CreateSingleChain
  function of the MCMCFactory class to create an instance of the MCMC algorithm we defined in the
  property tree.
  */
  //
  //auto mcmc = MCMCFactory::CreateSingleChain(pt, problem);
  Eigen::VectorXd startPt = mu;

  Eigen::VectorXd mu_prop(2);
  mu_prop << 4.0, 2.0;

  Eigen::MatrixXd cov_prop(2,2);
  cov_prop << 1.0, 0.8,
         0.8, 1.5;
  auto proposalDensity = std::make_shared<Gaussian>(mu_prop, cov_prop);

  auto proposal = std::make_shared<MCSampleProposal>(pt, problem, proposalDensity);

  std::vector<std::shared_ptr<TransitionKernel> > kernels = {std::make_shared<DummyKernel>(pt, problem, proposal)};

  auto mcmc = std::make_shared<SingleChainMCMC>(pt, kernels);

  /***
  ### 3. Run the MCMC algorithm
  We are now ready to run the MCMC algorithm.  Here we start the chain at the
  target densities mean.   The resulting samples are returned in an instance
  of the SampleCollection class, which internally holds the steps in the MCMC chain
  as a vector of weighted SamplingState's.
  */
  mcmc->Run(startPt);
  std::shared_ptr<SampleCollection> samps = mcmc->GetQOIs();

  /***
  ### 4. Analyze the results

  When looking at the entries in a SampleCollection, it is important to note that
  the states stored by a SampleCollection are weighted even in the MCMC setting.
  When a proposal $x^\prime$ is rejected, instead of making a copy of $x^{(k-1)}$
  for $x^{(k)}$, the weight on $x^{(k-1)}$ is simply incremented.  This is useful
  for large chains in high dimensional parameter spaces, where storing all duplicates
  could quickly consume available memory.

  The SampleCollection class provides several functions for computing sample moments.
  For example, here we compute the mean, variance, and third central moment.
  While the third moment is actually a tensor, here we only return the marginal
  values, i.e., $\mathbb{E}_x[(x_i-\mu_i)^3]$ for each $i$.
  */
  Eigen::VectorXd sampMean = samps->Mean();
  std::cout << "\nSample Mean = \n" << sampMean.transpose() << std::endl;

  Eigen::VectorXd sampVar = samps->Variance();
  std::cout << "\nSample Variance = \n" << sampVar.transpose() << std::endl;

  Eigen::MatrixXd sampCov = samps->Covariance();
  std::cout << "\nSample Covariance = \n" << sampCov << std::endl;

  Eigen::VectorXd sampMom3 = samps->CentralMoment(3);
  std::cout << "\nSample Third Moment = \n" << sampMom3 << std::endl << std::endl;

  return 0;
}
