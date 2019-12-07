import glob
import pdb
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
from huMoments import huMoments
from computeMHI import computeMHI


def normalized_euclidean(testMoments, trainMoments, variance, i):
    multiply = np.power(np.reshape(trainMoments[i, :], (-1, 1)) - testMoments, 2)
    divide = np.divide(multiply, variance)
    sum_up = np.sum(divide)
    result = np.sqrt(sum_up)
    return result

def showNearestMHI(testMoments, trainMoments, train_dir,K):
    train_set_len = trainMoments.shape[0]
    variance = np.nanvar(trainMoments, axis=0).reshape(-1, 1)
    # variance=variance.reshape(20, 1)
    distance = np.zeros((1, train_set_len))
    testMoments = np.reshape(testMoments, (-1, 1))
    # variance=variance.reshape(20, 1)
    nearest_MHI=np.zeros((480,640,K+1))
    for i in range(0, train_set_len):
        distance[:,i] = normalized_euclidean(testMoments,trainMoments,variance,i)
    sorted_index = np.argsort(distance)
    #predictedAction = np.int(trainLabels[sorted_index][0])
    #sorted directory
    shape = train_dir.shape
    sorted_directory=train_dir[sorted_index].reshape(shape)

    #find k image
    for i in range (0,K+1):
        print("compute MHI for ", i ,"Nearest ")
        nearest_MHI[:,:,i]= computeMHI(np.str(sorted_directory[i,:][0]))

    return nearest_MHI


if __name__ == "__main__":
    # GET ALLMHI
    basedir = './'
    actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']
    direct_list = []
    row = 480
    col = 640
    K=4
    for actionnum in range(len(actions)):
        subdirname = basedir + actions[actionnum] + '/'
        subdir = os.listdir(subdirname)
        subdir = np.sort(subdir)
        #print(subdir)
        for dir in subdir:
            #print(dir)
            dir_sub = subdirname + dir
            #print(dir_sub)
            tmp = np.array(dir_sub)
            direct_list = np.r_[direct_list, tmp]
    direct_list=np.asarray(direct_list[1:]).reshape(-1,1)
    #choose TRAIN
    trainMoments= np.load('huVectors.npy')
    trainMoments=np.array(trainMoments)

    # choose TEST
    testMoments = np.load('botharms-up-p1-1_Vector.npy', allow_pickle=True)
    Near_MHI = showNearestMHI(testMoments, trainMoments, direct_list, K)
    # plot the original
    figure=plt.figure(frameon=False)
    figure.suptitle("Nearest MHI for botharms-up-p1-1")
    for i in range(0, K + 1):
        if i == 0:
            im = Near_MHI[:, :, 0]
            ax = plt.subplot(2, 3, i + 1)
            ax.set_title("origin image")
            ax.imshow(im, cmap='gray')
        else:
            im = Near_MHI[:, :, i]
            ax = plt.subplot(2, 3, i + 1)
            ax.set_title("Nearest Neighbor" + str(i))
            ax.imshow(im, cmap='gray')
    figure.savefig("Nearest Neibor for botharm-up-p1-1")
    plt.show()

    '''
    # choose TEST
    MHI=computeMHI('./crouch/crouch-p2-1')
    testMoments = np.array(huMoments(MHI))

    #testMoments = np.load('crouch-up-p1-1_Vector.npy', allow_pickle=True)
    Near_MHI = showNearestMHI(testMoments, trainMoments, direct_list, K)
    # plot the original
    figure2 = plt.figure(frameon=False)
    figure2.suptitle("Nearest Neighbor for crouch-up-p2-1")
    #figure2.set_cmap('jet')
    for i in range(0, K + 1):
        if i == 0:
            im = Near_MHI[:, :, 0]
            ax = plt.subplot(2, 3, i + 1)
            ax.set_title("origin image")
            ax.imshow(im,cmap='gray')
        else:
            im = Near_MHI[:, :, i]
            ax = plt.subplot(2, 3, i + 1)
            ax.set_title("Nearest Neighbor" + str(i))
            ax.imshow(im,cmap='gray' )
    figure2.savefig("Nearest Neibor for crouch-p2-1")
    plt.show()
    '''
    '''
    # choose TEST
    testMoments = np.load('leftarmup-p1-1_Vector.npy', allow_pickle=True)
    Near_MHI = showNearestMHI(testMoments, trainMoments, direct_list, K)
    # plot the original
    figure = plt.figure(frameon=False)
    figure.suptitle("Nearest MHI for leftarm-up-p1-1")
    for i in range(0, K + 1):
        if i == 0:
            im = Near_MHI[:, :, 0]
            ax = plt.subplot(2, 3, i + 1)
            ax.set_title("origin image")
            ax.imshow(im, cmap='gray')
        else:
            im = Near_MHI[:, :, i]
            ax = plt.subplot(2, 3, i + 1)
            ax.set_title("Nearest Neighbor" + str(i))
            ax.imshow(im, cmap='gray')
    figure.savefig("Nearest Neibor for leftarmup-p1-2")
    plt.show()
    '''
