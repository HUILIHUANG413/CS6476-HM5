import glob
import pdb
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np

#function [predictedLabel] = predictAction(testMoments, trainMoments, trainLabels)
def euclidean_distance(trainMoments, testMoments, variance, i):
    multiply = np.power(np.reshape(trainMoments[i, :], (-1, 1)) - testMoments, 2)
    divide = np.divide(multiply, variance)
    sum_up = np.sum(divide)
    result = np.sqrt(sum_up)
    return result

def predictAction(testMoments, trainMoments, trainLabels):
    train_set_len=trainMoments.shape[0]
    variance=np.nanvar(trainMoments,axis=0).reshape(-1,1)
    #variance=variance.reshape(20, 1)
    distance=np.zeros((1,train_set_len))
    testMoments=np.reshape(testMoments, (-1,1))
    for i in range (0,train_set_len):
        distance[:,i]=euclidean_distance(trainMoments,testMoments,variance,i)
    sorted_index = np.argsort(distance)
    predictedAction = np.int(trainLabels[sorted_index][0,0])
    return predictedAction

if __name__ == "__main__":
    testMoments = np.load('botharms-up-p1-1_Vector.npy',allow_pickle=True)
    #testMoments = np.load('crouch-up-p1-1_Vector.npy',allow_pickle=True)
    #testMoments = np.load('rightkick_p1-1_Vector.npy',allow_pickle=True)
    trainMoments = np.asarray(np.load('huVectors.npy',allow_pickle=True))
    trainLabels = np.array([[1],[1],[1],[1],[2],[2],[2],[2],[3],[3],[3],[3],[4],[4],[4],[4],[5],[5],[5],[5]])
    predict_action=predictAction(testMoments,trainMoments,trainLabels)
    print(predict_action)

