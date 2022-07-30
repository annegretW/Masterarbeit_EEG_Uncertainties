#ifndef FLOWMODELMICOMPONENTS_HPP_
#define FLOWMODELMICOMPONENTS_HPP_

#include "MUQ/Utilities/MultiIndices/MultiIndex.h"

#include "MUQ/Modeling/LinearAlgebra/EigenLinearOperator.h"

#include "MUQ/SamplingAlgorithms/SamplingProblem.h"

#include "FlowEquation.h"

struct Discretization {
  Discretization(std::size_t const numCells);

  /// The number of cells in the mesh
  const std::size_t numCells;

  /// The number of nodes in the mesh
  const std::size_t numNodes;

  /// The node locations
  const Eigen::VectorXd nodeLocs;

  /// Cell locations (mid point of each cell)
  const Eigen::VectorXd cellLocs;
};

struct Data {
  Data(Eigen::VectorXd const& x, Eigen::VectorXd const& soln, Eigen::VectorXd const& obsLoc, Eigen::VectorXd const& obs);

  /// The locations where we have computed the true solution
  const Eigen::VectorXd x;

  /// The true solution
  const Eigen::VectorXd soln;

  /// The locations where we have made observations
  const Eigen::VectorXd obsLoc;

  /// The observations
  const Eigen::VectorXd obs;
};

/// Interpolate the model solution onto the observation points
/**
@param[in] mesh The mesh used to solve the model
@param[in] obsLoc The locations where we want are making observations
*/
std::shared_ptr<muq::Modeling::LinearOperator> InterpolateOntoObs(Discretization const& mesh, Eigen::VectorXd const& obsLoc);

/**
@param[in] mesh Evaluate the recharge (source terms) on the cell locations of this mesh
\return The recharge on the cell locations of this mesh
*/
Eigen::VectorXd GetRecharge(Discretization const& mesh);

/**
@param[in] mesh Evaluate the true log conductivity on the cell locations of this mesh
\return The true log conductivity on the cell locations of this mesh
*/
Eigen::VectorXd GetTrueLogConductivity(Discretization const& mesh);

/**
@param[in] numCells The number of cells on the very fine mesh used to generate the data
@param[in] numObs The number of observations
@param[in] obsVar The variance of the observation noise
\return The observations
*/
Data GenerateData(std::size_t const numCells, std::size_t const numObs, double const obsVar);

/**
@param[in] numLevels The number of multi level MCMC chains
@param[in] baseRefinement The number of cells in the coarsest model
*/
std::vector<std::shared_ptr<muq::Modeling::ModPiece> > ConstructDensities(std::size_t const numLevels, std::size_t const baseRefinement);

/**
@param[in] numCells The number of cells in the model associated with this density
@param[in] data The observations
@param[in] likelihoodVar The variance of the likelihood
*/
std::shared_ptr<muq::Modeling::ModPiece> ConstructDensity(std::size_t const numCells, Data const& data, double const likelihoodVar);

#endif
