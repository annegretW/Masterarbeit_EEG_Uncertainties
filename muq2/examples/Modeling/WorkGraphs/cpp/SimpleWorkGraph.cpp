#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/CwiseOperators/CwiseUnaryOperator.h"

using namespace muq::Modeling;

int main(){
  unsigned int dim = 2;
  auto f = std::make_shared<SinOperator>(dim);
  auto g = std::make_shared<ExpOperator>(dim);

  auto graph = std::make_shared<WorkGraph>();

  graph->AddNode(f,"f");
  graph->AddNode(g,"g");
  graph->AddEdge("f",0,"g",0); // connect output 0 of f with input 0 of g

  auto gof = graph->CreateModPiece("f");

  Eigen::VectorXd x = Eigen::VectorXd::Random(dim);
  std::cout << "exp(sin) = " << gof->Evaluate(x).at(0).transpose() << std::endl;
  return 0;
}
