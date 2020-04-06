import os
import sys
import cv2
import numpy as np
from NN import NN 
from NN import ACTIVATION
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def plotResult(SGD,miniBatch):
    x = np.array([25,50,100,150])
    plt.scatter(x,SGD,c='g',label='SDG')
    plt.scatter(x,miniBatch,c='r',label='mini Batch')

    plt.title('Accuracy of SDG vs mini Batch with 25 Hidden Neurons(sigmoid)')
    plt.xlabel('number of epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

def main():
    train = np.empty((1000,28,28),dtype='float64')
    trainY = np.zeros((1000,10,1))
    test = np.empty((10000,28,28),dtype='float64')
    testY = np.zeros((10000,10,1))

    # Load in the images
    i = 0
    for filename in os.listdir('C:/Users/anast/Documents/Deep Learning/Data/Training1000/'):
        y = int(filename[0])
        trainY[i,y] = 1.0
        train[i] = cv2.imread('C:/Users/anast/Documents/Deep Learning/Data/Training1000/{0}'.format(filename),0)/255.0 # 0 flag stands for greyscale; for color, use 1
        i = i + 1
    i = 0 # read test data
    for filename in os.listdir('C:/Users/anast/Documents/Deep Learning/Data/Test10000/'):
        y = int(filename[0])
        testY[i,y] = 1.0
        test[i] = cv2.imread('C:/Users/anast/Documents/Deep Learning/Data/Test10000/{0}'.format(filename),0)/255.0
        i = i + 1
    trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2],1)
    testX = test.reshape(test.shape[0],test.shape[1]*test.shape[2],1)

    activation=ACTIVATION.SIGMOID
    w1,b1,w2,b2=NN.trainSGD(trainX,trainY,activFunc=activation)
    accuracy= NN.getAccuracy(testX,testY,w1,b1,w2,b2,activFunc=activation)
    print("Accuracy of Classification is ",accuracy,'%') 
    w1,b1,w2,b2=NN.trainMiniBatch(trainX,trainY,activFunc=activation)
    accuracy= NN.getAccuracy(testX,testY,w1,b1,w2,b2,activFunc=activation)
    print("Accuracy of Classification is ",accuracy,'%')


    #graph accuracy over #epochs
    '''SGD=np.zeros((4,1))
    miniBatch=np.zeros((4,1))
    X = np.array([25,50,100,150])
    activation=ACTIVATION(1)
    for i in range(4):
        x =  X[i]
        w1,b1,w2,b2=NN.trainSGD(trainX,trainY,epochsNum=x,activFunc=activation)
        SGD[i]= NN.getAccuracy(testX,testY,w1,b1,w2,b2)
        w1,b1,w2,b2=NN.trainMiniBatch(trainX,trainY,epochsNum=x,activFunc=activation)
        miniBatch[i]= NN.getAccuracy(testX,testY,w1,b1,w2,b2)
    print("SGD: ",SGD)
    print("miniBatch: ",miniBatch)
    plotResult(SGD,miniBatch)'''

    
    #print("Accuracy of Classification is ",accuracy,'%')
    '''X=testX[0]
    print(X.shape)
    Y=testY[0]
    print(Y.shape)
    _,a=NN.forwardProp(w1,b1,w2,b2,X)
    print("y:", Y)
    print("a: ", a)'''

if __name__ == "__main__":
    sys.exit(int(main() or 0))

