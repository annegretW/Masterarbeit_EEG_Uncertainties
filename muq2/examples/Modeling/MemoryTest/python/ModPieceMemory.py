# Run This script with:
#   python -m memory_profiler ModPieceMemory.py

import muq.Modeling as mm

import numpy as np

import matplotlib.pyplot as plt

import psutil, os
process = psutil.Process(os.getpid())

from os import path

from memory_profiler import memory_usage
#from memory_profiler import profile

#import gc

class Test1(mm.PyModPiece):

    def __init__(self, dim):
        mm.PyModPiece.__init__(self, [dim],[dim]) # (inputSizes, outputSizes)


    def EvaluateImpl(self, inputs):

        mem = process.memory_info()
        self.outputs = [np.exp(inputs[0])]

class OtherModel:

    def __init__(self, dim):
        self.dim = dim

    def Evaluate(self,input):
        return [np.exp(input)]

#@profile
def PythonPiece():
    dim = 200

    input = np.ones(dim)
    mod = Test1(dim)

    for i in range(20000):
        res = mod.Evaluate([input])

    pass

#@profile
def CppPiece():
    dim = 200

    input = np.ones(dim)
    mod = mm.ExpOperator(dim)

    for i in range(20000):
        res = mod.Evaluate([input])

    pass

#@profile
def PurePython():
    dim = 200

    input = np.ones(dim)
    mod = OtherModel(dim)

    for i in range(20000):
        res = mod.Evaluate([input])

    pass

interval = 0.01
mem1 = memory_usage((PythonPiece,(),{}), interval=interval)
plt.plot(np.linspace(0,1,len(mem1)), [m - mem1[0] for m in mem1],'-+',label='Python ModPiece')

mem2 = memory_usage((CppPiece,(),{}), interval=interval)
plt.plot(np.linspace(0,1,len(mem2)), [m - mem2[0] for m in mem2],'-+',label='C++ ModPiece from Python')

mem3 = memory_usage((PurePython,(),{}), interval=interval)
plt.plot(np.linspace(0,1,len(mem3)), [m - mem3[0] for m in mem3],'-+',label='Pure Python Class')

cppFile = '../cpp/build/cpp_memory.txt'
if(path.exists(cppFile)):
    mem4 = np.loadtxt(cppFile)
    mem4 /= (1024*1024)
    plt.plot(np.linspace(0,1,len(mem4)), mem4-mem4[0],'-',label='C++ ModPiece from C++')

plt.xlabel('Normalized Time (start->finish)')
plt.ylabel('Memory Use [MB]')
plt.legend()
plt.show()

#CppPiece()
#sPurePython()
#mem1 = memory_usage((PythonPiece,(),{}))
#mem2 = memory_usage((CppPiece,(),{}))
#mem3 = memory_usage((PurePython,(),{}))
#print(mem1)
#print(mem2)
#print(mem3)
