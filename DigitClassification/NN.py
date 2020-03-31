import numpy as np
class NN(object):
       
    def trainNN(X,Y):
        w1 = np.random.uniform(-0.12,0.12,(100,784))
        b1 = np.random.uniform(-0.12,0.12,(100,1))
        w2 = np.random.uniform(-0.12,0.12,(10,100))
        b2 = np.random.uniform(-0.12,0.12,(10,1))
        epochs=100
        #for i in range(epochs):
        return w1,b1,w2,b2
    


