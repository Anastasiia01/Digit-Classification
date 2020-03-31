import numpy as np
from sklearn.utils import shuffle

class NN(object):
       
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def forwardProp(w1,b1,w2,b2,xi):
        #w1 is 100x784, b1 is 100x1, xi is 784x1, w2 is 10x100, b2 is 10x1
        s1=w1@xi+b1 #100x1
        a1=NN.sigmoid(s1) #100x1
        s2=w2@a1+b2 #10x1
        a2=NN.sigmoid(s2)
        return s1,a1,s2,a2

    def computeLoss(actualOut,expectedOut):
        #should return total loss of current model 
        x=0

    def derivativeOfLoss(t):
        #should return partial derivative of t with respect to loss
        x=0

    def trainNN(X,Y,alpha=0.1): #X is (1000x784x1), Y is (1000x10x1)
        w1 = np.random.uniform(-0.12,0.12,(100,784))
        b1 = np.random.uniform(-0.12,0.12,(100,1))
        w2 = np.random.uniform(-0.12,0.12,(10,100))
        b2 = np.random.uniform(-0.12,0.12,(10,1))
        #regularization parameter:lambda
        regLambda=0.01
        epochsNum=1 #100
        samplesNum=5#X.shape[0]
        newX=X
        newY=Y

        for j in range(epochsNum):
            newX,newY=shuffle(newX,newY)
            for i in range(samplesNum):
                xi=X[i]#same as X[i,:]
                #compute forward pass
                s1,a1,s2,a2 = NN.forwardProp(w1,b1,w2,b2,xi)
                '''loss=computeLoss(a2,Y[i])
                backpropagation: updating weigths and biases
                w1=w1-alpha(derivativeOfLoss(w1)- regLambda*w1)
                b1=b1-alpha(derivativeOfLoss(b1)- regLambda*b1)
                w2=w2-alpha(derivativeOfLoss(w2)- regLambda*w2)
                b2=b2-alpha(derivativeOfLoss(b2)- regLambda*b2)'''

        return w1,b1,w2,b2    


    


