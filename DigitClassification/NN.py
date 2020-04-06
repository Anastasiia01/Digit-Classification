import numpy as np
from sklearn.utils import shuffle
from enum import Enum

class ACTIVATION(Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3

class NN(object):      
    
    def activationFunc(x,activFunc):
        if(activFunc.name=='SIGMOID'):
            return 1/(1+np.exp(-x))
        elif(activFunc.name=='TANH'):
            return np.tanh(x)
        else:#RELU y=x if x>0 and y=0 if x<=0
            der=np.copy(x)
            der[x<=0]=0
            return der

    def forwardProp(w1,b1,w2,b2,xi,activFunc):
        #w1 is 100x784, b1 is 100x1, xi is 784x1, w2 is 10x100, b2 is 10x1
        s1=w1@xi+b1 #100x1
        a1=NN.activationFunc(s1, activFunc) #100x1
        s2=w2@a1+b2 #10x1
        a2=np.exp(s2)/np.sum(np.exp(s2))#softmax as output activation function
        #a2=NN.activationFunc(s2, ACTIVATION(1))#sigmoid as output activation function
        return a1,a2

    def derOvera1(a1, activFunc):
        if(activFunc.name=='SIGMOID'):
            return a1*(1-a1)
        elif(activFunc.name=='TANH'):
            return (1-a1**2)
        else:#RELU' y=1 if x>0 and y=0 if x<=0
            der=np.zeros(a1.shape)
            der[a1>0]=1
            return der

    def getAccuracy(X,Y,w1,b1,w2,b2,activFunc=ACTIVATION(1)):
        rightCount=0
        testsNum=X.shape[0]
        for i in range(testsNum):
            xi=X[i]
            yi=Y[i]
            _,a2=NN.forwardProp(w1,b1,w2,b2,xi,activFunc)
            maxIdx=np.argmax(a2)
            if(yi[maxIdx]==1):
                rightCount+=1
        accuracy=(rightCount/testsNum)*100
        return accuracy

                #Returns the indices of the min values along an axis.

    def computeLoss(a2,yi,w1,w2):
        reglambda=0.01
        Idx=np.argmax(yi)
        loss=-np.sum(np.log(a2[Idx]))#cross entropy loss for softmax
        #loss=(((a2-yi)*(a2-yi))).sum()#MSE loss for sigmoid
        loss+=reglambda / 2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
        return loss
      

    def computeDerivatives(xi,yi,a2,a1,w2,w1,activFunc=ACTIVATION(1)):
        #s2 and a2 are 10x1, s1 and a1 are 100x1, xi is 784x1, yi is 10x1
        #should return partial derivatives of w1,b1,w2,b2 with respect to loss
        # w2 is 10x100, b2 is 10x1, delta1 is 100x1, w1 is 100x784,b1 is 100x1
        #delta2=(a2-yi)*NN.derOvera1(a2, ACTIVATION(1))#for sigmoid as activation func of output layer
        delta2=a2-yi#with softmax as act.func of output layer
        dw2=delta2@a1.T #10x100
        db2=delta2 #delta2*1
        delta1=(w2.T@delta2)*NN.derOvera1(a1, activFunc)
        dw1=delta1@xi.T
        db1=delta1 #delta1*1

        # Add regularization terms (b1 and b2 don't have regularization terms)
        reglambda=0.01
        dw2+=reglambda*w2
        dw1+=reglambda*w1
        return dw1,db1,dw2,db2

    def trainSGD(X,Y,alpha=0.01,epochsNum=100,activFunc=ACTIVATION(1)): #X is (1000x784x1), Y is (1000x10x1)
        numNeuronsLayer1 = 30
        numNeuronsLayer2 = 10
        w1 = np.random.uniform(-0.1,0.1,(numNeuronsLayer1,784))
        b1 = np.random.uniform(-0.1,0.1,(numNeuronsLayer1,1))
        w2 = np.random.uniform(-0.1,0.1,(numNeuronsLayer2,numNeuronsLayer1))
        b2 = np.random.uniform(-0.1,0.1,(numNeuronsLayer2,1))
        samplesNum=X.shape[0]

        for j in range(epochsNum):
            X,Y=shuffle(X,Y)
            loss=0
            for i in range(samplesNum):
                xi=X[i]#same as X[i,:]
                yi=Y[i]
                #compute forward pass
                a1,a2 = NN.forwardProp(w1,b1,w2,b2,xi,activFunc)
                loss+=NN.computeLoss(a2,yi,w1,w2)
                #backpropagation: updating weigths and biases
                dw1,db1,dw2,db2=NN.computeDerivatives(xi,yi,a2,a1,w2,w1,activFunc)
                w1=w1-alpha*(dw1)
                b1=b1-alpha*(db1)
                w2=w2-alpha*(dw2)
                b2=b2-alpha*(db2)
            #print("epoch = ", j, "loss = ", loss)
        return w1,b1,w2,b2    

    def trainMiniBatch(X,Y,alpha=0.01,epochsNum=100,batchSize=10,activFunc=ACTIVATION(1)): #X is (1000x784x1), Y is (1000x10x1)
        numNeuronsLayer1 = 100
        numNeuronsLayer2 = 10
        w1 = np.random.uniform(-0.1,0.1,(numNeuronsLayer1,784))
        b1 = np.random.uniform(-0.1,0.1,(numNeuronsLayer1,1))
        w2 = np.random.uniform(-0.1,0.1,(numNeuronsLayer2,numNeuronsLayer1))
        b2 = np.random.uniform(-0.1,0.1,(numNeuronsLayer2,1))
        samplesNum=X.shape[0]
        for j in range(epochsNum):
            X,Y=shuffle(X,Y)
            loss=0
            for k in range(0,samplesNum-batchSize+1,batchSize):
                avgdw1=0
                avgdb1=0
                avgdw2=0
                avgdb2=0
                for i in range(batchSize):
                    xi=X[k+i]#same as X[k+i,:]
                    yi=Y[k+i]
                    #compute forward pass
                    a1,a2 = NN.forwardProp(w1,b1,w2,b2,xi,activFunc)
                    loss+=NN.computeLoss(a2,yi,w1,w2)
                    dw1,db1,dw2,db2=NN.computeDerivatives(xi,yi,a2,a1,w2,w1,activFunc)
                    avgdw1+=dw1
                    avgdb1+=db1
                    avgdw2+=dw2
                    avgdb2+=db2
                avgdw1/=batchSize
                avgdb1/=batchSize
                avgdw2/=batchSize
                avgdb2/=batchSize
                #backpropagation: updating weigths and biases after every batch
                w1=w1-alpha*(avgdw1)
                b1=b1-alpha*(avgdb1)
                w2=w2-alpha*(avgdw2)
                b2=b2-alpha*(avgdb2)
            #print("epoch = ", j, "loss = ", loss)
        return w1,b1,w2,b2    


    


