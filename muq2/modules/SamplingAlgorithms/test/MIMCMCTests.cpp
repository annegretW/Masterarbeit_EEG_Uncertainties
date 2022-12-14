#include "MUQ/SamplingAlgorithms/SLMCMC.h"
#include "MUQ/SamplingAlgorithms/GreedyMLMCMC.h"
#include "MUQ/SamplingAlgorithms/MIMCMC.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"

#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"

#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"

#include <boost/property_tree/ptree.hpp>

#include <gtest/gtest.h>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;



class MySamplingProblem : public AbstractSamplingProblem {
public:
  MySamplingProblem(std::shared_ptr<muq::Modeling::ModPiece> targetIn)
   : AbstractSamplingProblem(Eigen::VectorXi::Constant(1,2), Eigen::VectorXi::Constant(1,2)),
     target(targetIn){}

  virtual ~MySamplingProblem() = default;


  virtual double LogDensity(std::shared_ptr<SamplingState> const& state) override {
    lastState = state;
    return target->Evaluate(state->state).at(0)(0);
  };

  virtual std::shared_ptr<SamplingState> QOI() override {
    assert (lastState != nullptr);
    return std::make_shared<SamplingState>(lastState->state, 1.0);
  }

private:
  std::shared_ptr<SamplingState> lastState = nullptr;

  std::shared_ptr<muq::Modeling::ModPiece> target;

};


class MyInterpolation : public MIInterpolation {
public:
  std::shared_ptr<SamplingState> Interpolate (std::shared_ptr<SamplingState> const& coarseProposal, std::shared_ptr<SamplingState> const& fineProposal) {
    return std::make_shared<SamplingState>(coarseProposal->state);
  }
};

class MyMIComponentFactory : public MIComponentFactory {
public:

  MyMIComponentFactory(pt::ptree pt) : pt(pt) {}

  virtual std::shared_ptr<MCMCProposal> Proposal (std::shared_ptr<MultiIndex> const& index, std::shared_ptr<AbstractSamplingProblem> const& samplingProblem) override {
    pt::ptree pt;
    pt.put("BlockIndex",0);

    Eigen::VectorXd mu(2);
    mu << 1.0, 2.0;
    Eigen::MatrixXd cov(2,2);
    cov << 0.7, 0.6,
    0.6, 1.0;
    cov *= 20.0;

    auto prior = std::make_shared<Gaussian>(mu, cov);

    return std::make_shared<CrankNicolsonProposal>(pt, samplingProblem, prior);
  }

  virtual std::shared_ptr<MultiIndex> FinestIndex() override {
    auto index = std::make_shared<MultiIndex>(2);
    index->SetValue(0, 2);
    index->SetValue(1, 2);
    return index;
  }

  virtual std::shared_ptr<MCMCProposal> CoarseProposal (std::shared_ptr<MultiIndex> const& fineIndex,
                                                        std::shared_ptr<MultiIndex> const& coarseIndex,
                                                        std::shared_ptr<AbstractSamplingProblem> const& coarseProblem,
                                                        std::shared_ptr<SingleChainMCMC> const& coarseChain) override {
    pt::ptree ptProposal = pt;
    ptProposal.put("BlockIndex",0);
    return std::make_shared<SubsamplingMIProposal>(ptProposal, coarseProblem, coarseIndex, coarseChain);
  }

  virtual std::shared_ptr<AbstractSamplingProblem> SamplingProblem (std::shared_ptr<MultiIndex> const& index) override {
    Eigen::VectorXd mu(2);
    mu << 1.0, 2.0;
    Eigen::MatrixXd cov(2,2);
    cov << 0.7, 0.6,
           0.6, 1.0;

    if (index->GetValue(0) == 0) {
      mu[0] *= 0.8;
    } else if (index->GetValue(0) == 1) {
      mu[0] *= 0.9;
    } else if (index->GetValue(0) == 2) {
      mu[0] *= 1.0;
    } else {
      std::cerr << "Sampling problem not defined!" << std::endl;
      assert (false);
    }
    if (index->GetValue(1) == 0) {
      mu[1] *= 0.8;
    } else if (index->GetValue(1) == 1) {
      mu[1] *= 0.9;
    } else if (index->GetValue(1) == 2) {
      mu[1] *= 1.0;
    } else {
      std::cerr << "Sampling problem not defined!" << std::endl;
      assert (false);
    }
    if (index->Max() == 0) {
      cov *= 2.0;
    } else if (index->Max() == 1) {
      cov *= 1.5;
    } else if (index->Max() == 2) {
      cov *= 1.0;
    } else {
      std::cerr << "Sampling problem not defined!" << std::endl;
      assert (false);
    }

    auto coarseTargetDensity = std::make_shared<Gaussian>(mu, cov)->AsDensity();
    return std::make_shared<MySamplingProblem>(coarseTargetDensity);
  }

  virtual std::shared_ptr<MIInterpolation> Interpolation (std::shared_ptr<MultiIndex> const& index) override {
    return std::make_shared<MyInterpolation>();
  }

  virtual Eigen::VectorXd StartingPoint (std::shared_ptr<MultiIndex> const& index) override {
    Eigen::VectorXd mu(2);
    mu << 1.0, 2.0;
    return mu;
  }

private:
  pt::ptree pt;
};

TEST(MIMCMCTest, MIMCMC) {

  pt::ptree pt;
  pt.put("NumSamples_0_0", 1e3);
  pt.put("NumSamples_0_1", 1e3);
  pt.put("NumSamples_0_2", 1e3);
  pt.put("NumSamples_1_0", 1e3);
  pt.put("NumSamples_1_1", 1e3);
  pt.put("NumSamples_1_2", 1e3);
  pt.put("NumSamples_2_0", 1e3);
  pt.put("NumSamples_2_1", 1e3);
  pt.put("NumSamples_2_2", 1e3);
  pt.put("MLMCMC.Subsampling_0_0", 5);
  pt.put("MLMCMC.Subsampling_0_1", 5);
  pt.put("MLMCMC.Subsampling_0_2", 5);
  pt.put("MLMCMC.Subsampling_1_0", 5);
  pt.put("MLMCMC.Subsampling_1_1", 5);
  pt.put("MLMCMC.Subsampling_1_2", 5);
  pt.put("MLMCMC.Subsampling_2_0", 5);
  pt.put("MLMCMC.Subsampling_2_1", 5);
  pt.put("MLMCMC.Subsampling_2_2", 5);

  auto componentFactory = std::make_shared<MyMIComponentFactory>(pt);

  MIMCMC mimcmc (pt, componentFactory);
  mimcmc.Run();
  mimcmc.Draw(false);

  auto samps = mimcmc.GetSamples();
  auto mean = samps->Mean();
  Eigen::VectorXd mcse = samps->StandardError();

  std::cout << "MIMCMC MCSE: " << mcse.transpose() << std::endl;
  EXPECT_NEAR(1.0, mean(0), 3.0*mcse(0));
  EXPECT_NEAR(2.0, mean(1), 3.0*mcse(1));

  auto qois = mimcmc.GetQOIs();
  mean = qois->Mean();
  mcse = qois->StandardError();

  std::cout << "MIMCMC MCSE: " << mcse.transpose() << std::endl;
  EXPECT_NEAR(1.0, mean(0), 3.0*mcse(0));
  EXPECT_NEAR(2.0, mean(1), 3.0*mcse(1));

}

TEST(MIMCMCTest, SLMCMC)
{

  pt::ptree pt;

  pt.put("NumSamples", 5e3); // number of samples for single level

  auto componentFactory = std::make_shared<MyMIComponentFactory>(pt);

  SLMCMC slmcmc (pt, componentFactory);
  slmcmc.Run();

  auto mean = slmcmc.GetQOIs()->Mean();

  EXPECT_NEAR(mean[0], 1.0, 0.1);
  EXPECT_NEAR(mean[1], 2.0, 0.1);

}
