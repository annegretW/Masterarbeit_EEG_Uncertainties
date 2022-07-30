#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/SumPiece.h"
#include "MUQ/Modeling/SplitVector.h"
#include "MUQ/Modeling/CwiseOperators/CwiseUnaryOperator.h"

using namespace muq::Modeling;

int main(){
  auto f = std::make_shared<SinOperator>(2);
  auto g = std::make_shared<ExpOperator>(2);
  auto sum = std::make_shared<SumPiece>(2);

  // Will split x_{1:dim} into two equally sized vectors
  auto splitter = std::make_shared<SplitVector>(std::vector<int>{0,2}, // indices of output
                                                std::vector<int>{2,2}, // sizes of output
                                                4); // size of input

  auto graph = std::make_shared<WorkGraph>();

  graph->AddNode(splitter, "x12,x34");
  graph->AddNode(g,"g");
  graph->AddNode(f,"f");
  graph->AddEdge("x12,x34",0,"f",0); // connect output 0 of x12,x34 with input 0 of f
  graph->AddEdge("x12,x34",0,"g",0); // connect output 1 of x12,x34 with input 0 of g

  graph->AddNode(sum,"f+g");
  graph->AddEdge("f",0,"f+g",0); // connect output 0 of f with input 0 of f+g
  graph->AddEdge("g",0,"f+g",1); // connect output 0 of g with intpu 1 of f+g

  auto mod = graph->CreateModPiece("f+g");

  Eigen::VectorXd x = Eigen::VectorXd::Random(4);
  std::cout << "result = " << mod->Evaluate(x).at(0).transpose() << std::endl;
  return 0;
}
