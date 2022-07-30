import muq.Modeling as mm
import numpy as np

dim = 2
f = mm.SinOperator(dim)
g = mm.ExpOperator(dim)

graph = mm.WorkGraph()
graph.AddNode(f,"f")
graph.AddNode(g,"g")
graph.AddEdge("f",0,"g",0) # <- connect output 0 of f with input 0 of g

gof = graph.CreateModPiece("f")

x = np.random.randn(dim)
print("exp(sin) = ", gof.Evaluate([x])[0] )
