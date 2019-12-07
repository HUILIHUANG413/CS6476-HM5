import glob
import pdb
import os
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

threshold=39100
def computeMHI(directoryName):
    depthfiles = glob.glob(directoryName + '/' + '*.pgm');
    print(depthfiles)
    depthfiles = np.sort(depthfiles)
    tau=len(depthfiles)
    img=imread(depthfiles[0])
    row=img.shape[0]
    col=img.shape[1]
    Motion_History_Image=np.zeros((row,col))
    for ith_image in range (0, tau):
        #print(str(ith_image + 1) +" out of " + str(tau) )
        #read image
        img=imread(depthfiles[ith_image])
        #image pre-analysis
        img[img < threshold]=1
        img[img > threshold]=0
        if ith_image > 0 :
            #compute difference between two image
            dif_img=np.absolute(pre_img-img)
            #compute MHI
            for i in range (row):
                for j in range(col):
                    if (dif_img[i][j] == 1):
                        Motion_History_Image[i][j]=tau
                    else:
                        tmp=max(0,Motion_History_Image[i][j]-1)
                        Motion_History_Image[i][j]=tmp
            pre_img=img
        else:
            pre_img=img
    #normalization
    Motion_History_Image=Motion_History_Image/np.max(Motion_History_Image)
    return Motion_History_Image

if __name__ == "__main__":

    #Save three images
    # both arm up

    Motion_History_img = computeMHI('./botharms/botharms-up-p1-1')
    row=Motion_History_img.shape[0]
    col=Motion_History_img.shape[1]

    plt.figure()
    plt.set_cmap('jet')
    plt.imshow(Motion_History_img)
    plt.title("both arms")
    plt.show()
    #plt.savefig("botharms-up-p1-1_MHI.png")
    #np.save("botharms-up-p1-1_MHI.npy",Motion_History_img)
    
    #Crouch
    Motion_History_img = computeMHI('./crouch/crouch-p1-1')
    plt.figure()
    plt.set_cmap('jet')
    plt.imshow(Motion_History_img)
    plt.title("crouch")
    plt.savefig('crouch-p1-1_MHI.png')
    np.save("crouch-p1-1_MHI.npy", Motion_History_img)
    # Crouch
    Motion_History_img = computeMHI('./rightkick/rightkick-p1-1')
    plt.figure()
    plt.set_cmap('jet')
    plt.imshow(Motion_History_img)
    plt.title("rightkick")
    plt.savefig('rightkick-p1-1_MHI.png')
    np.save("rightkick-p1-1_MHI.npy", Motion_History_img)

    Motion_History_img = computeMHI('./rightkick/rightkick-p1-2')
    plt.figure()
    plt.set_cmap('jet')
    plt.imshow(Motion_History_img)
    plt.title("rightkick")
    plt.savefig('rightkick-p1-2_MHI.png')
    np.save("rightkick-p1-2_MHI.npy", Motion_History_img)
    Motion_History_img = computeMHI('./rightkick/rightkick-p2-1')

    plt.figure()
    plt.set_cmap('jet')
    plt.imshow(Motion_History_img)
    plt.title("rightkick")
    plt.savefig('rightkick-p2-1_MHI.png')
    np.save("rightkick-p2-1_MHI.npy", Motion_History_img)


    # GET ALLMHI
    basedir = './'
    actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']
    direct_list= []
    row=480
    col=640
    all_MHI=np.zeros((row,col,20))

    for actionnum in range(len(actions)):
        subdirname = basedir + actions[actionnum] + '/'
        subdir = os.listdir(subdirname)
        subdir = np.sort(subdir)
        print(subdir)
        for dir in subdir:
            print(dir)
            dir_sub= subdirname + dir
            print(dir_sub)
            tmp=np.array(dir_sub)
            direct_list=np.r_[direct_list,tmp]
    print(direct_list)
    # loop through each subdirectory
    for i in range(1, len(direct_list)):
        print("the direct is: ", direct_list[i])
        all_MHI[:, :, i-1] = computeMHI(np.str(direct_list[i]))
    
    np.save('allMHIs.npy', all_MHI)
















