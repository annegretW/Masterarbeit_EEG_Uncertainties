#include <vector>
#include <Eigen/Core>

#include "MUQ/Modeling/ModPiece.h"


class SimpleModel : public muq::Modeling::ModPiece
{
public:
  SimpleModel(int numPts) : muq::Modeling::ModPiece({numPts,2},{numPts}) {};

protected:
  void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    Eigen::VectorXd y = c(0)*x + c(1)*Eigen::VectorXd::Ones(x.size());

    outputs.resize(1);
    outputs.at(0) = y;
  };

  virtual void JacobianImpl(unsigned int outWrt,
                            unsigned int inWrt,
                            muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Jacobian wrt x
    if(inWrt==0){
      jacobian = c(0)*Eigen::VectorXd::Identity(x.size(), x.size());

    // Jacobian wrt c
    }else{
      jacobian = Eigen::MatrixXd::Ones(outputSizes(0), inputSizes(inWrt));
      jacobian.col(0) = x;
    }
  }

  virtual void GradientImpl(unsigned int outWrt,
                            unsigned int inWrt,
                            muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                            Eigen::VectorXd const& sens) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Gradient wrt x
    if(inWrt==0){
      gradient = c(0) * sens;

    // Gradient wrt c
    }else{
      gradient.resize(2);
      gradient(0) = x.dot(sens);
      gradient(1) = sens.sum();
    }
  }

  virtual void ApplyJacobianImpl(unsigned int outWrt,
                                 unsigned int inWrt,
                                 muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                 Eigen::VectorXd const& vec) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Jacobian wrt x
    if(inWrt==0){
      jacobianAction = c(0)*vec;

    // Jacobian wrt c
    }else{
      jacobianAction = vec(0)*x + vec(1)*Eigen::VectorXd::Ones(x.size());
    }
  }

  virtual void ApplyHessianImpl(unsigned int outWrt,
                                 unsigned int inWrt1,
                                 unsigned int inWrt2,
                                 muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                 Eigen::VectorXd const& sens,
                                 Eigen::VectorXd const& vec) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Apply d^2 / dxdc
    if((inWrt1==0)&&(inWrt2==1)){
      hessAction = vec(0) * sens;

    // Apply d^2 / dcdx
    }else if((inWrt2==0)&&(inWrt1==1)){
      hessAction.resize(2);
      hessAction(0) = sens.dot(vec);
      hessAction(1) = 0;

    // Apply d^2 / dxds
    }else if((inWrt1==0)&&(inWrt2==2)){
      hessAction = c(0) * vec;

    // Apply d^2 / dcds
    }else if((inWrt1==1)&&(inWrt2==2)){

      hessAction.resize(2);
      hessAction(0) = x.dot(vec);
      hessAction(1) = vec.sum();

    // Apply d^2/dx^2  or  d^2/dc^2  or  d^2/ds^2 or d^2 / dsdx or  d^2 / dsdc
    }else{
      hessAction = Eigen::VectorXd::Zero(inputSizes(inWrt1));
    }
  }
}; // end of class SimpleModel


int main(){

  unsigned int numPts = 10;
  Eigen::VectorXd x = Eigen::VectorXd::Random(numPts);

  Eigen::VectorXd c(2);
  c << 1.0, 0.5;

  auto mod = std::make_shared<SimpleModel>(numPts);

  Eigen::VectorXd y = mod->Evaluate(x,c).at(0);

  std::cout << "c = " << c.transpose() << std::endl;
  std::cout << "x = " << x.transpose() << std::endl;
  std::cout << "y = " << y.transpose() << std::endl;

  int wrt1=0, wrt2=0;

  Eigen::VectorXd grad, gradFd, vec;
  Eigen::VectorXd hessAct, hessActFd;
  Eigen::VectorXd sens = Eigen::VectorXd::Random(mod->outputSizes(0));


  grad = mod->Gradient(0,wrt1, x,c,sens);
  gradFd = mod->GradientByFD(0,wrt1,std::vector<Eigen::VectorXd>{x,c},sens);
  std::cout << "Gradient Comparison:\n" << grad.transpose() << "\nvs\n" << gradFd.transpose() << std::endl << std::endl;

  vec = Eigen::VectorXd::Random(mod->inputSizes(wrt2));
  hessAct = mod->ApplyHessian(0, wrt1, wrt2, std::vector<Eigen::VectorXd>{x,c}, sens, vec);
  hessActFd = mod->ApplyHessianByFD(0,wrt1,wrt2,std::vector<Eigen::VectorXd>{x,c},sens,vec);

  std::cout << "Hessian Comparison:\n" << hessAct.transpose() << "\nvs\n" << hessActFd.transpose() << std::endl << std::endl;

  wrt1 = 1;
  grad = mod->Gradient(0,wrt1, x,c,sens);
  gradFd = mod->GradientByFD(0,wrt1,std::vector<Eigen::VectorXd>{x,c},sens);
  std::cout << "Gradient Comparison:\n" << grad.transpose() << "\nvs\n" << gradFd.transpose() << std::endl << std::endl;

  vec = Eigen::VectorXd::Random(mod->inputSizes(wrt2));
  hessAct = mod->ApplyHessian(0, wrt1, wrt2, std::vector<Eigen::VectorXd>{x,c}, sens, vec);
  hessActFd = mod->ApplyHessianByFD(0,wrt1,wrt2,std::vector<Eigen::VectorXd>{x,c},sens,vec);

  std::cout << "Hessian Comparison:\n" << hessAct.transpose() << "\nvs\n" << hessActFd.transpose() << std::endl << std::endl;

  wrt1 = 0;
  wrt2 = 1;

  grad = mod->Gradient(0,wrt1, x,c,sens);
  gradFd = mod->GradientByFD(0,wrt1,std::vector<Eigen::VectorXd>{x,c},sens);
  std::cout << "Gradient Comparison:\n" << grad.transpose() << "\nvs\n" << gradFd.transpose() << std::endl << std::endl;

  vec = Eigen::VectorXd::Random(mod->inputSizes(wrt2));
  hessAct = mod->ApplyHessian(0, wrt1, wrt2, std::vector<Eigen::VectorXd>{x,c}, sens, vec);
  hessActFd = mod->ApplyHessianByFD(0,wrt1,wrt2,std::vector<Eigen::VectorXd>{x,c},sens,vec);

  std::cout << "Hessian Comparison:\n" << hessAct.transpose() << "\nvs\n" << hessActFd.transpose() << std::endl << std::endl;

  wrt2 = 2;

  vec = Eigen::VectorXd::Random(mod->outputSizes(0));
  hessAct = mod->ApplyHessian(0, wrt1, wrt2, std::vector<Eigen::VectorXd>{x,c}, sens, vec);
  hessActFd = mod->ApplyHessianByFD(0,wrt1,wrt2,std::vector<Eigen::VectorXd>{x,c},sens,vec);

  std::cout << "Hessian Comparison:\n" << hessAct.transpose() << "\nvs\n" << hessActFd.transpose() << std::endl << std::endl;


  return 0;
}
