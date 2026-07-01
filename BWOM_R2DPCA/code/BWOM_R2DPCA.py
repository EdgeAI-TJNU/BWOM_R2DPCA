import numpy as np
import cv2
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# L1 norm 2DPCA based on binary weighting and optimal mean - Greedy Solution
train_path = "D:\\DataSet\\jxy\\COIL-100\\COIL-100\\20\\train1\\%d.png"
test_path = "D:\\DataSet\\jxy\\COIL-100\\COIL-100\\test\\%d.png"
trainlable = "D:\\DataSet\\jxy\\COIL-100\\COIL-100\\train.txt"
testlable = "D:\\DataSet\\jxy\\COIL-100\\COIL-100\\test.txt"

# save_path = "D:\\DataSet\\cg\\Our\\COIL\\%d.jpg"#reconstructed image save path

num_train = 700
num_test = 700
m = 100  # rows ORL:112 PIE:64
n = 100  # cols  ORL:92 PIE:64
K = 50
beta = 0.8  #Assume that the proportion of normal samples, denoted as c/N in the paper
c = np.ceil(num_train * beta)  # Assume the number of normal samples, which refers to the number of '1' elements in the weight vector.

x_train = np.zeros([num_train, m, n], np.float32)
x_train_c = np.zeros([num_train, m, n], np.float32)
x_test = np.zeros([num_test, m, n], np.float32)
x = np.zeros([m, n], np.float32)
err_ = []
for i in range(0, num_train):
    image = cv2.imread(train_path % i, cv2.IMREAD_GRAYSCALE)
    x_train[i, :, :] = image
    x += image

A = np.zeros([num_train, m, n], np.float32)
W = np.zeros([n, K], np.float32)
W[0:K, 0:K] = np.eye(K)
P = np.ones([num_train])
OM = x / num_train
for ss in range(0, 10):
    print("Number of Iterations：", ss + 1)
    Wp = np.copy(W)
    Pp = np.copy(P)
    OMp = np.copy(OM)
    for i in range(0, num_train):
        A[i, :, :] = x_train[i, :, :] - OMp

    # Step 1: Optimize W
    for k in range(0, K, 1):
        print("   ", k + 1)

        w = W[:, k]
        while True:
            wp = w
            # Calculate the Gradient
            v = np.zeros([n, ], np.float32)
            for i in range(0, num_train):
                v += P[i] * A[i, :, :].T @ np.sign(A[i, :, :] @ w)
            w = (np.abs(v) * np.sign(v)) / (np.linalg.norm(v, ord=2))

            # Calculate the Judging Condition
            F = 0
            for i in range(0, num_train):
                F += np.linalg.norm(A[i, :, :] @ w - A[i, :, :] @ wp, ord=2) / np.linalg.norm(
                    A[i, :, :] @ wp, ord=2)
            # print(F)
            if F <= 1e-4:
                break
        for i in range(0, num_train):
            A[i, :, :] = A[i, :, :] - A[i, :, :] @ np.dot(w.reshape(-1, 1), w.reshape(1, -1))
        W[:, k] = w

    # Step 2: Optimize Binary Weights P
    P = np.zeros([num_train], np.float32)
    E = np.zeros([num_train], np.float32)
    for i in range(num_train):
        E[i] = np.linalg.norm((x_train[i, :, :] - OMp) - (x_train[i, :, :] - OMp) @ W @ W.T)
    ind = np.argsort(E)  # ind is the index sorted in ascending order. The reconstruction errors are sorted from smallest to largest, and the first c samples are considered normal samples, while the rest are considered noise samples.
    for i in range(num_train):
        if i < c:
            P[ind[i]] = 1
        else:
            P[ind[i]] = 0
    # Sample Weight
    # print(P)

    # Step 3: Optimize the Optimal Weight Mean
    OM = np.zeros([m, n], np.float32)
    for i in range(num_train):
        OM += P[i] * x_train[i, :, :]
    OM = OM / np.count_nonzero(P)

    print("norm(ΔP)", np.linalg.norm(P - Pp, ord=2))
    print("norm(ΔM)", np.linalg.norm(OM - OMp, ord=2))
    # Convergence Judgment (The judging criteria can be modified as needed)
    if (np.linalg.norm(P - Pp, ord=2) <= 2) & (np.linalg.norm(OM - OMp, ord=2) <= 5):
        break

# Performance Testing
#for k in range(10, K + 1, 10):
for k in range(1, K + 1, 1):  #Change the step size to 1 here.
    Wk = W[:, 0:k]
    for i in range(0, num_test):
        image = cv2.imread(test_path % i, cv2.IMREAD_GRAYSCALE)
        x_test[i, :, :] = image - OM

    # label
    ltrain = np.loadtxt(trainlable)
    ltest = np.loadtxt(testlable)
    train_data = np.zeros((num_train, k * m))
    test_data = np.zeros((num_test, k * m))
    for i in range(0, num_train):
        train_data[i, :] = np.dot(x_train[i, :, :] - OM, Wk).reshape(1, -1)
    for i in range(0, num_test):
        test_data[i, :] = np.dot(x_test[i, :, :], Wk).reshape(1, -1)

    # SVM
    acc = 0
    #svm_classifier = svm.SVC(kernel='rbf')
    svm_classifier = svm.SVC(kernel='rbf')
    svm_classifier.fit(train_data, ltrain)
    label_test = svm_classifier.predict(test_data)
    for i in range(0, num_test):
        if label_test[i] == ltest[i]:
            acc = acc + 1
    acc = acc / num_test * 100
    print(acc)

    # # KNN
    # acc = 0
    # knn = KNeighborsClassifier(n_neighbors=1)
    # knn.fit(train_data, ltrain)
    # label_test = knn.predict(test_data)
    # for i in range(0, num_test):
    #    if label_test[i] == ltest[i]:
    #        acc = acc + 1
    # acc = acc / num_test * 100
    # print(acc)

    # Reconstruction Error
    err = 0
    for i in range(0, num_test):
        err += np.linalg.norm(x_test[i, :, :] - x_test[i, :, :] @ Wk @ Wk.T, ord='fro')
    err = err / num_test
    err_.append(err)

print(np.array(err_).reshape(-1))

# # Performance Testing
# # Reconstructed Image  K is the order of the principal components to be retained.
# for k in range(K, K + 1, 1):
#     Wk = W[:, 0:k]
#     for i in range(0, num_test):
#         image = cv2.imread(test_path % i, cv2.IMREAD_GRAYSCALE)
#         x_test[i, :, :] = image - OM
#
#     for i in range(0, num_test):
#         recon_image = x_test[i, :, :] @ Wk @ Wk.T + OM
#         cv2.imwrite(save_path % i, recon_image)
