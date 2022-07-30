import muq.Modeling as mm
import numpy as np

class SimpleModel(mm.PyModPiece):

    def __init__(self, numPts):
        super(SimpleModel,self).__init__([numPts,2],[numPts])

    def EvaluateImpl(self, inputs):
        x,c  = inputs

        y = c[0]*x + c[1]

        self.outputs = [y]

    def JacobianImpl(self, outWrt, inWrt, inputs):
        x,c = inputs

        # Jacobian wrt x
        if(inWrt==0):
            self.jacobian = c[0]*np.eye(x.shape[0])

        # Jacobian wrt c
        else:
            self.jacobian =np.ones((self.outputSizes[0], self.inputSizes[inWrt]))
            self.jacobian[:,0] = x

    def GradientImpl(self, outWrt, inWrt, inputs, sens):
        x, c = inputs

        # Gradient wrt x
        if(inWrt==0):
            self.gradient = c[0] * sens

        # Gradient wrt c
        else:
            self.gradient = np.array([x.dot(sens),sens.sum()])

    def ApplyJacobianImpl(outWrt, inWrt, inputs, vec):
        x,c = inputs

        # Jacobian wrt x
        if(inWrt==0):
            self.jacobianAction = c[0]*vec

        # Jacobian wrt c
        else:
            self.jacobianAction = vec[0]*x + vec[1]*np.ones(x.shape[0])

    def ApplyHessianImpl(outWrt, inWrt1, inWrt2, inputs, sens, vec):
        x,c = inputs

        # Apply d^2 / dxdc
        if((inWrt1==0)&(inWrt2==1)):
            hessAction = vec[0] * sens

        # Apply d^2 / dcdx
        elif((inWrt2==0)&(inWrt1==1)):
            hessAction = np.array([sens.dot(vec),0])

        # Apply d^2 / dxds
        elif((inWrt1==0)&(inWrt2==2)):
            hessAction = c[0]*vec

        # Apply d^2 / dcds
        elif((inWrt1==1)&(inWrt2==2)):
            hessAction = np.array([x.dot(vec), vec.sum()])

        # Apply d^2/dx^2  or  d^2/dc^2  or  d^2/ds^2 or d^2 / dsdx or  d^2 / dsdc
        else:
            hessAction = np.zeros(self.inputSizes[inWrt1])


numPts = 10
x = np.random.randn(numPts)
c = np.array([1.0,0.5])

mod = SimpleModel(numPts)

y = mod.Evaluate(x,c)[0]

print('c = ', c)
print('x = ', x)
print('y = ', y)
