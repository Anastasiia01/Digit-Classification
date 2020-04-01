import os
import sys
import cv2
import numpy as np
from NN import NN 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def plotResult(SGD,miniBatch):
    x = np.arange(25, 101, 25)
    plt.plot(x,SGD,linewidth=0.8,c='g',label='SDG')
    plt.plot(x,miniBatch,linewidth=0.8,c='r',label='mini Batch')

    plt.title('Accuracy of SDG vs mini Batch with 25 Hidden Neurons')
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
    SGD=np.zeros((4,1))
    miniBatch=np.zeros((4,1))
    for i in range(4):
        x=(i+1)*25
        w1,b1,w2,b2=NN.trainSGD(trainX,trainY,epochsNum=x)
        SGD[i]= NN.getAccuracy(testX,testY,w1,b1,w2,b2)
        w1,b1,w2,b2=NN.trainMiniBatch(trainX,trainY,epochsNum=x)
        miniBatch[i]= NN.getAccuracy(testX,testY,w1,b1,w2,b2)
    plotResult(SGD,miniBatch)
    '''x = np.arange(25, 101, 25)
    plt.plot(x1,SGD,linewidth=0.8,c='g',label='SDG')
    plt.plot(x1,miniBatch,linewidth=0.8,c='r',label='mini Batch')

    plt.title('Accuracy of SDG vs mini Batch with 25 Hidden Neurons')
    plt.xlabel('number of epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()'''




    
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

