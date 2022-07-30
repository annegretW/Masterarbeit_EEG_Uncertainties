#include "MUQ/Modeling/CwiseOperators/CwiseUnaryOperator.h"

#include "mymemory.h"

#include <iostream>
#include <fstream>

using namespace muq::Modeling;


int main(){

  int dim = 200;
  auto mod = std::make_shared<ExpOperator>(dim);

  int numCalls = 20000;
  std::vector<int> mem(numCalls);

  Eigen::VectorXd input = Eigen::VectorXd::Ones(dim);
  std::vector<Eigen::VectorXd> res;
  for(int i=0; i<numCalls; ++i){
    res = mod->Evaluate(input);
    mem.at(i) = getCurrentRSS( );
  }

  // Write the memory usage to a file
  std::ofstream fout("cpp_memory.txt");
  for(int i=0; i<numCalls; ++i)
    fout << mem.at(i) << "\n";
  fout.close();

  return 0;
}
