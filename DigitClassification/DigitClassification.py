import os
import sys
import cv2
import numpy as np
from NN import NN 
from sklearn.utils import shuffle


def main():
    '''check how shuffle works
    X=np.array([[[0],[0],[3],[4],[3],[4],[3],[0],[0]],[[0],[0],[5],[4],[3],[4],[3],[0],[0]],[[0],[0],[6],[4],[3],[4],[3],[0],[0]],[[0],[0],[2],[4],[3],[4],[3],[0],[0]],[[0],[0],[9],[4],[3],[4],[3],[0],[0]]])
    print(X.shape)
    Y=np.zeros((5,10,1))
    Y[0,3]=1
    Y[1,5]=1
    Y[2,6]=1
    Y[3,2]=1
    Y[4,9]=1
    for i in range(5):
        X,Y=shuffle(X,Y)
        print(X)'''
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
    w1,b1,w2,b2=NN.trainNN(trainX,trainY)
    '''X=testX[0]
    print(X.shape)
    Y=testY[0]
    print(Y.shape)
    _,a=NN.forwardProp(X,w1,b1,w2,b2)
    print("y:", Y)
    print("a: ", a)'''

if __name__ == "__main__":
    sys.exit(int(main() or 0))

