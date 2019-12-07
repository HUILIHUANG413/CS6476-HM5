import numpy as np


def euclidean_distance(trainMoments, testMoments, variance, i):
    multiply = np.power(np.reshape(trainMoments[i, :], (-1, 1)) - testMoments, 2)
    divide = np.divide(multiply, variance)
    sum_up = np.sum(divide)
    result = np.sqrt(sum_up)
    return result

def classifyAllAction(testMoments, trainMoments, trainLabels,K):
    train_set_len = trainMoments.shape[0]
    variance = np.nanvar(trainMoments, axis=0).reshape(-1, 1)
    # variance=variance.reshape(20, 1)
    distance = np.zeros((1, train_set_len))
    testMoments = np.reshape(testMoments, (-1, 1))
    for i in range(0, train_set_len):
        distance[:, i] = euclidean_distance(trainMoments, testMoments, variance, i)
    sorted_index = np.argsort(distance)
    bin_count = np.bincount(trainLabels[sorted_index].flatten()[0:K])
    predictedAction = np.argmax(bin_count)
    return predictedAction

if __name__ == "__main__":
    trainMoments = np.asarray(np.load('huVectors.npy',allow_pickle=True))
    trainLabels = np.array([[1],[1],[1],[1],[2],[2],[2],[2],[3],[3],[3],[3],[4],[4],[4],[4],[5],[5],[5],[5]])
    #trainLabels = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])
    K=4
    predict_action=np.zeros(len(trainLabels))
    mean_recognition_rate=np.zeros(5)
    confusion_matrix= np.zeros((5,5))
    row,col=trainMoments.shape
    print("now compute the predict action: ")
    for i in range (0,row):
        testMoments=trainMoments[i,:]
        train_Moments=np.delete(trainMoments,i,0)
        train_Labels=np.delete(trainLabels,i,0)

        predict_action[i]=classifyAllAction(testMoments,train_Moments,train_Labels,K)

    print("now compute the confusion_matrix")
    for i in range (0,row):
        actual_class=np.int(trainLabels[i])
        pre_class=np.int(predict_action[i])
        confusion_matrix[actual_class-1,pre_class-1] += 1

    print("now compute the mean recognition rate per class")
    for i in range (0, 5):
        mean_recognition_rate[i]=confusion_matrix[i,i]/4

    overall_recognition_rate=np.mean(mean_recognition_rate)

    np.save("confusion_matrix.npy", confusion_matrix)
    np.save("mean_recognition_rate.npy", mean_recognition_rate)

    print("confusion_matrix",confusion_matrix)
    print("mean_recognition_rate",mean_recognition_rate)
    print("overall_recognition_rate:",overall_recognition_rate)
