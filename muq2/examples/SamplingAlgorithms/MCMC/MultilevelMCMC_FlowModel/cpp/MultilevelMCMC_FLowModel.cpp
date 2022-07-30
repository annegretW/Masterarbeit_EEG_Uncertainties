#include <iostream>

#include "MUQ/SamplingAlgorithms/GreedyMLMCMC.h"

#include "FlowModelMIComponents.h"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;

int main() {
  const std::size_t numLevels = 2;
  const std::size_t baseRefinement = 25; // The number of cells in the coarsest model
  std::vector<std::shared_ptr<ModPiece> > logDensities = ConstructDensities(numLevels, baseRefinement);

  pt::ptree options;

  options.put("NumInitialSamples", 1000); // number of initial samples for greedy MLMCMC
  options.put("GreedyTargetVariance", 0.05); // Target estimator variance to be achieved by greedy algorithm
  options.put("verbosity", 1); // show some output
  options.put("MLMCMC.Subsampling_0", 8);
  options.put("MLMCMC.Subsampling_1", 4);
  options.put("MLMCMC.Subsampling_2", 2);
  options.put("MLMCMC.Subsampling_3", 0);

  options.put("Proposal.Method", "MHProposal");
  options.put("Proposal.ProposalVariance", 1.0);

  Eigen::VectorXd theta0 = Eigen::VectorXd::Zero(logDensities[0]->inputSizes(0));
  GreedyMLMCMC sampler(options, theta0, logDensities);

  std::shared_ptr<MultiIndexEstimator> estimator = sampler.Run();

  std::cout << "Mean:     " << estimator->Mean().transpose() << std::endl;
  std::cout << "Variance: " << estimator->Variance().transpose() << std::endl;

  const std::string filename = "output.h5";
  sampler.WriteToFile(filename);

  /*// Define the mesh
  unsigned int numCells = 100;
  Discretization mesh(numCells);

  // Generate synthetic "truth" data
  unsigned int obsThin = 10;
  double obsVar = 0.01*0.01;
  auto data = GenerateData(mesh, obsThin, obsVar);

  auto posterior = DefinePosterior(mesh, data, obsThin, obsVar);

  // Using the "truth" as a starting point for MCMC
  Eigen::VectorXd startPt = GetTrueLogConductivity(mesh);*/
}
