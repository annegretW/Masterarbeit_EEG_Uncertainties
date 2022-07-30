import muq.Modeling as mm
import numpy as np

f = mm.SinOperator(2)
g = mm.ExpOperator(2)
sum = mm.SumPiece(2)

# Will split x_{1:dim} into two equally sized vectors
splitter = mm.SplitVector([0,2], # indices of output
                          [2,2], # sizes of output
                          4)     # size of input

graph = mm.WorkGraph()

graph.AddNode(splitter, "x12,x34");
graph.AddNode(g,"g")
graph.AddNode(f,"f")
graph.AddEdge("x12,x34",0,"f",0) # connect output 0 of x12,x34 with input 0 of f
graph.AddEdge("x12,x34",0,"g",0) # connect output 1 of x12,x34 with input 0 of g

graph.AddNode(sum,"f+g");
graph.AddEdge("f",0,"f+g",0) # connect output 0 of f with input 0 of f+g
graph.AddEdge("g",0,"f+g",1) # connect output 0 of g with intpu 1 of f+g

mod = graph.CreateModPiece("f+g")

x = np.random.randn(4)
print("result = ", mod.Evaluate([x])[0])
