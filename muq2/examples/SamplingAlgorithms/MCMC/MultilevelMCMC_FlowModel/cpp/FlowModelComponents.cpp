#include "FlowModelMIComponents.h"

#include "MUQ/Utilities/HDF5/HDF5File.h"

#include "MUQ/Modeling/LinearAlgebra/IdentityOperator.h"
#include "MUQ/Modeling/CwiseOperators/CwiseUnaryOperator.h"
#include "MUQ/Modeling/Distributions/DensityProduct.h"

#include "MUQ/Approximation/Polynomials/Legendre.h"
#include "MUQ/Approximation/Quadrature/GaussQuadrature.h"
#include "MUQ/Approximation/GaussianProcesses/SquaredExpKernel.h"
#include "MUQ/Approximation/GaussianProcesses/KarhunenLoeveExpansion.h"

using namespace muq::Utilities;
using namespace muq::Modeling;
using namespace muq::Approximation;
using namespace muq::SamplingAlgorithms;

Discretization::Discretization(std::size_t const numCells) :
numCells(numCells),
numNodes(numCells+1),
nodeLocs(Eigen::VectorXd::LinSpaced(numCells+1, -1.0, 1.0)),
cellLocs(0.5*(nodeLocs.tail(numCells) + nodeLocs.head(numCells)))
{}

Data::Data(Eigen::VectorXd const& x, Eigen::VectorXd const& soln, Eigen::VectorXd const& obsLoc, Eigen::VectorXd const& obs) :
x(x),
soln(soln),
obsLoc(obsLoc),
obs(obs)
{}

Eigen::VectorXd GetTrueLogConductivity(Discretization const& mesh) {
    return (M_PI*mesh.cellLocs).array().cos() + (3.0*M_PI*mesh.cellLocs).array().sin();
}

Eigen::VectorXd GetRecharge(Discretization const& mesh) {
  return Eigen::VectorXd::Ones(mesh.cellLocs.size());
}

std::shared_ptr<LinearOperator> InterpolateOntoObs(Discretization const& mesh, Eigen::VectorXd const& obsLoc) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(obsLoc.size(), mesh.numNodes);
  std::size_t j = 0;
  for( std::size_t i=0; i<obsLoc.size(); ++i ) {
    while( mesh.nodeLocs(j+1)<obsLoc(i) ) { ++j; }
    assert(j<mesh.numNodes-1);
    assert(obsLoc(i)>mesh.nodeLocs(j)-1.0e-14);
    assert(obsLoc(i)<mesh.nodeLocs(j+1)+1.0e-14);

    A(i, j) = (mesh.nodeLocs(j+1)-obsLoc(i))/(mesh.nodeLocs(j+1)-mesh.nodeLocs(j));
    assert(A(i, j)>-1.0e-14); assert(A(i, j)<1.0+1.0e-14);
    A(i, j+1) = 1.0-A(i, j);
  }

  return LinearOperator::Create(A);
}

Data GenerateData(std::size_t const numCells, std::size_t const numObs, double const obsVar) {
  const Discretization mesh(numCells);

  const Eigen::VectorXd trueCond = GetTrueLogConductivity(mesh).array().exp();
  const Eigen::VectorXd recharge = GetRecharge(mesh);

  auto mod = std::make_shared<FlowEquation>(recharge);
  const Eigen::VectorXd obsLoc = Eigen::VectorXd::LinSpaced(numObs, -1.0, 1.0);
  auto interp = InterpolateOntoObs(mesh, obsLoc);

  // solve the forward problem with the true conductivity
  const Eigen::VectorXd trueSol = mod->Evaluate(trueCond).at(0);
  return Data(mesh.nodeLocs, trueSol, obsLoc, interp->Evaluate(trueSol).at(0) + std::sqrt(obsVar)*RandomGenerator::GetNormal(interp->outputSizes(0)));
}

std::shared_ptr<ModPiece> ConstructDensity(std::size_t const numCells, Data const& data, double const likelihoodVar) {
  const Discretization mesh(numCells);

  const double sigma2 = 1.0;
  const double length = 0.1;
  auto kernel = std::make_shared<SquaredExpKernel>(1, sigma2, length);

  GaussQuadrature quad(std::make_shared<Legendre>());
  const std::size_t order = 15;
  quad.Compute(order);
  const Eigen::RowVectorXd points = quad.Points();
  const Eigen::VectorXd weights = quad.Weights();

  KarhunenLoeveExpansion klexpansion(kernel, points, weights);
  const Eigen::Matrix modes = klexpansion.GetModes(mesh.cellLocs.transpose());
  auto logConductivity = LinearOperator::Create(modes);

  auto prior = std::make_shared<Gaussian>(modes.cols());

  const std::size_t numSamps = 15;
  Eigen::MatrixXd priorSamples(numSamps, mesh.numCells);
  for( std::size_t i=0; i<numSamps; ++i ) {
    priorSamples.row(i) = logConductivity->Evaluate(prior->Sample()).at(0);
  }

  const std::string filename = "output.h5";
  auto file = std::make_shared<HDF5File>(filename);
  file->WriteMatrix("/prior information/x", mesh.cellLocs);
  file->WriteMatrix("/prior information/samples", priorSamples);
  file->Close();

  auto graph = std::make_shared<WorkGraph>();
  graph->AddNode(std::make_shared<IdentityOperator>(logConductivity->inputSizes(0)), "theta");
  graph->AddNode(logConductivity, "log conductivity");
  graph->AddEdge("theta", 0, "log conductivity", 0);

  graph->AddNode(std::make_shared<ExpOperator>(mesh.numCells), "conductivity");
  graph->AddEdge("log conductivity", 0, "conductivity", 0);

  // define the forward model
  Eigen::VectorXd recharge = GetRecharge(mesh);
  auto forwardMod = std::make_shared<FlowEquation>(recharge);

  graph->AddNode(forwardMod, "forward model");
  graph->AddEdge("conductivity", 0, "forward model", 0);

  graph->AddNode(InterpolateOntoObs(mesh, data.obsLoc), "observations");
  graph->AddEdge("forward model", 0, "observations", 0);

  auto likelihood = std::make_shared<Gaussian>(data.obs, likelihoodVar*Eigen::VectorXd::Ones(data.obs.size()));
  graph->AddNode(likelihood->AsDensity(), "likelihood");
  graph->AddEdge("observations", 0, "likelihood", 0);

  graph->AddNode(prior->AsDensity(), "prior");
  graph->AddEdge("theta", 0, "prior", 0);

  graph->AddNode(std::make_shared<DensityProduct>(2), "posterior");
  graph->AddEdge("prior", 0, "posterior", 0);
  graph->AddEdge("likelihood", 0, "posterior", 1);

  graph->Visualize("WorkGraph.pdf");

  return graph->CreateModPiece("posterior");
}

std::vector<std::shared_ptr<ModPiece> > ConstructDensities(std::size_t const numLevels, std::size_t const baseRefinement) {
  // generate the data
  const std::size_t numObs = 25;
  const double obsVar = 1.0e-4;
  const Data data = GenerateData(baseRefinement*(numLevels+1), numObs, obsVar);

  const std::string filename = "output.h5";
  auto file = std::make_shared<HDF5File>(filename);
  file->WriteMatrix("/truth/x", data.x);
  file->WriteMatrix("/truth/solution", data.soln);
  file->WriteMatrix("/data/locations", data.obsLoc);
  file->WriteMatrix("/data/observations", data.obs);
  file->Close();

  std::vector<std::shared_ptr<ModPiece> > logDensities(numLevels);

  for( std::size_t i=0; i<numLevels; ++i ) {
    logDensities[i] = ConstructDensity(baseRefinement*(i+1), data, obsVar);
  }

  return logDensities;
}
