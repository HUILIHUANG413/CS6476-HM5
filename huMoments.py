import glob
import pdb
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import computeMHI


def huMoments(H):
    H = np.array(H)
    row=H.shape[0]
    col=H.shape[1]

    #create meshgrid
    x = np.arange (0, col, 1)
    y = np.arange (0, row, 1)
    X,Y=np.meshgrid(x,y)
    X=X+1
    Y=Y+1
    X=np.array(X)
    Y=np.array(Y)

    M10=np.sum(X*H)
    M00=np.sum(H)
    M01=np.sum(Y * H)

    #calculate central
    x_bar = np.float(M10/M00)
    y_bar = np.float(M01/M00)

    u02 = np.sum(np.power(X - x_bar, 0) * np.power(Y - y_bar, 2) * H)
    u02= np.sum(np.power((X-x_bar),0) * np.power((Y-y_bar),2) * H, dtype=np.float64)
    u03 = np.sum(np.power(X-x_bar,0)* np.power(Y-y_bar,3) * H)
    u11 = np.sum(np.power(X - x_bar, 1) * np.power(Y - y_bar, 1) * H)
    u12 = np.sum(np.power(X - x_bar, 1) * np.power(Y - y_bar, 2) * H)
    u20 = np.sum(np.power(X - x_bar, 2) * np.power(Y - y_bar, 0) * H)
    u21 = np.sum(np.power(X - x_bar, 2) * np.power(Y - y_bar, 1) * H)
    u30 = np.sum(np.power(X - x_bar, 3) * np.power(Y - y_bar, 0) * H)

    #calculate Hu moments
    h1 = u20 + u02
    h2 = np.power((u20 - u02), 2) + 4 * np.power(u11, 2)
    h3 = np.power((u30 - 3 * u12), 2) + np.power((3 * u21 - u03), 2)
    h4 = np.power((u30 + u12), 2) + np.power((u21 + u03), 2)
    h5 = (u30 - 3 * u12) * (u30 + u12) * (np.power((u30 + u12), 2) - 3 * np.power((u21 + u03), 2)) + (
                3 * u21 - u03) * (u21 + u03) * (3 * np.power((u30 + u12), 2) - np.power((u21 + u03), 2))
    h6 = (u20 - u02) * (np.power((u30 + u12), 2) - np.power((u21 + u03), 2)) + 4 * u11 * (
                u30 + u12) * (u21 + u03)
    h7 = (3 * u21 - u03) * (u30 + u12) * (np.power((u30 + u12), 2) - 3 * np.power((u21 + u03), 2)) - (
                u30 - 3 * u12) * (u21 + u03) * (3 * np.power((u30 + u12), 2) - np.power((u21 + u03), 2))

    Hu_moments=[h1,h2,h3,h4,h5,h6,h7]

    return Hu_moments

if __name__ == "__main__":


    MHI_kick_test=np.load('rightkick-p1-1_MHI.npy')
    rightkick_Vector=huMoments(MHI_kick_test)
    np.save('rightkick_p1-1_Vector.npy',rightkick_Vector)

    MHI_kick_test = np.load('rightkick-p2-1_MHI.npy')
    rightkick_Vector = huMoments(MHI_kick_test)
    np.save('rightkick_p2-1_Vector.npy', rightkick_Vector)

    MHI_kick_test = np.load('rightkick-p1-2_MHI.npy')
    rightkick_Vector = huMoments(MHI_kick_test)
    np.save('rightkick_p1-2_Vector.npy', rightkick_Vector)

    MHI_arm_test = np.load('botharms-up-p1-1_MHI.npy')
    arm_Vector = huMoments(MHI_arm_test)
    np.save('botharms-up-p1-1_Vector.npy', arm_Vector)
    MHI_crouch_test = np.load('crouch-p1-1_MHI.npy')
    crouch_Vector = huMoments(MHI_crouch_test)
    np.save('crouch-up-p1-1_Vector.npy', crouch_Vector)


    allMHI=np.load('allMHIs.npy')
    row=allMHI.shape[0]
    col=allMHI.shape[1]

    huVectors=[]
    for i in range (20):
        print("i=",i)
        tmp=huMoments(allMHI[:,:,i])
        huVectors.append(tmp)
    huVectors=np.array(huVectors)

    motion_history_images = np.load('allMHIs.npy')

    # initlize variables
    nth_row, nth_column, number_of_motition_history_images = motion_history_images.shape
    huVectors = np.zeros((number_of_motition_history_images, 7))

    # calculate hu vector for each motion history image and store them as one hu vectors
    for ith_motion_history_image in range(0, number_of_motition_history_images):
        huVectors[ith_motion_history_image, :] = huMoments(motion_history_images[:, :, ith_motion_history_image])

    np.save('huVectors.npy',huVectors)



