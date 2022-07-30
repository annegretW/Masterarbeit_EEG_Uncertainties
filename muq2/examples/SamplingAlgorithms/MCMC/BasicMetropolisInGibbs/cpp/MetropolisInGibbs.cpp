#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/Modeling/Distributions/DensityProduct.h"
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/ModGraphPiece.h"

#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/Diagnostics.h"

#include "MUQ/Utilities/RandomGenerator.h"

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

int main(){

  Eigen::VectorXd mux(1), varx(1);
  mux << 1.0;
  varx << 0.5;

  auto px = std::make_shared<Gaussian>(mux,varx)->AsDensity();

  Eigen::VectorXd muy(2), vary(2);
  muy << -1.0, -1.0;
  vary << 1.0, 2.0;

  auto py = std::make_shared<Gaussian>(muy,vary)->AsDensity();

  auto graph = std::make_shared<WorkGraph>();
  graph->AddNode(px, "p(x)");
  graph->AddNode(py, "p(y)");
  graph->AddNode(std::make_shared<DensityProduct>(2), "p(x,y)");
  graph->AddEdge("p(x)",0,"p(x,y)",0);
  graph->AddEdge("p(y)",0,"p(x,y)",1);

  auto pxy = graph->CreateModPiece("p(x,y)");

  // Define the sampling problem as normal
  auto problem = std::make_shared<SamplingProblem>(pxy);

  // Construct two kernels: one for x and one for y
  boost::property_tree::ptree opts;

  // A vector to holding the two transition kernels
  std::vector<std::shared_ptr<TransitionKernel>> kernels(2);

  // Construct the kernel on x
  opts.put("ProposalVariance", 3.0);
  opts.put("BlockIndex", 0); // Specify that this proposal should target x
  auto propx = std::make_shared<MHProposal>(opts, problem);
  kernels.at(0) = std::make_shared<MHKernel>(opts, problem, propx);

  // Construct the kernel on y
  opts.put("ProposalVariance", 5.0);
  opts.put("BlockIndex", 1); // Specify that this proposal should target y
  auto propy = std::make_shared<MHProposal>(opts, problem);
  kernels.at(1) = std::make_shared<MHKernel>(opts, problem, propy);

  // Construct the MCMC sampler using this transition kernel
  opts.put("NumSamples", 2000);
  opts.put("BurnIn", 100);
  opts.put("PrintLevel", 3);


  // Run 4 independent chains to help assess convergence
  unsigned int numChains = 4;
  std::vector<std::shared_ptr<SampleCollection>> chains(numChains);
  std::vector<Eigen::VectorXd> startPt(2);

  for(int i=0; i<numChains;++i){
    startPt.at(0) = RandomGenerator::GetNormal(1);
    startPt.at(1) = 2.0*RandomGenerator::GetNormal(2); // Initial point for y

    auto sampler = std::make_shared<SingleChainMCMC>(opts, kernels);
    chains.at(i) = sampler->Run(startPt);
  }

  // Compute the Rhat convergence diagnostic
  Eigen::VectorXd rhat = Diagnostics::Rhat(chains);
  std::cout << "Rhat = " << rhat.transpose() << std::endl;

  // Estimate the total effective sample size
  Eigen::VectorXd ess = chains.at(0)->ESS();
  for(int i=1; i<numChains; ++i)
    ess += chains.at(i)->ESS();

  std::cout << "ESS: " << ess.transpose() << std::endl;



  return 0;
}
