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
        return a1,a2

    def computeLoss(a2,yi):
        loss=(0.5*((a2-yi)*(a2-yi))).sum()
        return loss
      

    def computeDerivatives(xi,yi,a2,a1,w2):
        #s2 and a2 are 10x1, s1 and a1 are 100x1, xi is 784x1, yi is 10x1
        #should return partial derivatives of w1,b1,w2,b2 with respect to loss
        # w2 is 10x100, b2 is 10x1, delta1 is 100x1, w1 is 100x784,b1 is 100x1
        delta2=(a2-yi)*a2*(1-a2)#delta2 is 10x1
        dw2=delta2@a1.T #10x100
        db2=delta2 #delta2*1
        delta1=(w2.T@delta2)*a1*(1-a1)
        dw1=delta1@xi.T
        db1=delta1 #delta1*1
        return dw1,db1,dw2,db2

    def trainNN(X,Y,alpha=0.1): #X is (1000x784x1), Y is (1000x10x1)
        w1 = np.random.uniform(-0.12,0.12,(100,784))
        b1 = np.random.uniform(-0.12,0.12,(100,1))
        w2 = np.random.uniform(-0.12,0.12,(10,100))
        b2 = np.random.uniform(-0.12,0.12,(10,1))
        #regularization parameter:lambda
        epochsNum=50
        samplesNum=X.shape[0]
        for j in range(epochsNum):
            X,Y=shuffle(X,Y)
            loss=0
            for i in range(samplesNum):
                xi=X[i]#same as X[i,:]
                yi=Y[i]
                #compute forward pass
                a1,a2 = NN.forwardProp(w1,b1,w2,b2,xi)
                loss+=NN.computeLoss(a2,yi)
                dw1,db1,dw2,db2=NN.computeDerivatives(xi,yi,a2,a1,w2)
                #backpropagation: updating weigths and biases
                w1=w1-alpha*(dw1)
                b1=b1-alpha*(db1)
                w2=w2-alpha*(dw2)
                b2=b2-alpha*(db2)
            print("epoch= ", j, "loss= ", loss)
        return w1,b1,w2,b2    


    


