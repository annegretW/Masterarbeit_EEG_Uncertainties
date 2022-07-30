#include "MUQ/SamplingAlgorithms/ParallelizableMIComponentFactory.h"

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
  MyMIComponentFactory (Eigen::VectorXd const& start, pt::ptree pt)
   : startingPoint(start), pt(pt)
  { }

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
    auto index = std::make_shared<MultiIndex>(1);
    index->SetValue(0, 3);
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
      mu *= 0.8;
      cov *= 2.0;
    } else if (index->GetValue(0) == 1) {
      mu *= 0.9;
      cov *= 1.5;
    } else if (index->GetValue(0) == 2) {
      mu *= 0.99;
      cov *= 1.1;
    } else if (index->GetValue(0) == 3) {
      mu *= 1.0;
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
    return startingPoint;
  }

private:
  Eigen::VectorXd startingPoint;
  pt::ptree pt;
};


