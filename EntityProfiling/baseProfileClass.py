import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import sys
import warnings

warnings.filterwarnings('ignore')


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")


## -- 1 -- ##
def plot3Classes(data1, name1, data2, name2, data3, name3):
    plt.subplot(3, 1, 1)
    plt.plot(data1)
    plt.title(name1)
    plt.subplot(3, 1, 2)
    plt.plot(data2)
    plt.title(name2)
    plt.subplot(3, 1, 3)
    plt.plot(data3)
    plt.title(name3)
    plt.show()
    waitforEnter()


## -- 2 -- ##
def breakTrainTest(data, oWnd=300, trainPerc=0.5):
    nSamp, nCols = data.shape
    nObs = int(nSamp / oWnd)
    data_obs = data[:nObs * oWnd, :].reshape((nObs, oWnd, nCols))

    order = np.random.permutation(nObs)
    order = np.arange(nObs)  # Comment out to random split

    nTrain = int(nObs * trainPerc)

    data_train = data_obs[order[:nTrain], :, :]
    data_test = data_obs[order[nTrain:], :, :]

    return data_train, data_test


## -- 3 -- ##
def extractFeatures(data, Class=0):
    features = []
    nObs, nSamp, nCols = data.shape
    oClass = np.ones((nObs, 1)) * Class
    for i in range(nObs):
        M1 = np.mean(data[i, :, :], axis=0)
        # Md1=np.median(data[i,:,:],axis=0)
        Std1 = np.std(data[i, :, :], axis=0)
        # S1=stats.skew(data[i,:,:])
        K1 = stats.kurtosis(data[i, :, :])
        p = [75, 90, 95]
        Pr1 = np.array(np.percentile(data[i, :, :], p, axis=0)).T.flatten()

        # faux=np.hstack((M1,Md1,Std1,S1,K1,Pr1))
        faux = np.hstack((M1, Std1, K1, Pr1))
        features.append(faux)

    return np.array(features), oClass


## -- 4 -- ##
def plotFeatures(features, oClass, f1index=0, f2index=1):
    nObs, nFea = features.shape
    colors = ['b', 'g', 'r']
    for i in range(nObs):
        plt.plot(features[i, f1index], features[i, f2index], 'o' + colors[int(oClass[i])])

    plt.show()
    waitforEnter()


def logplotFeatures(features, oClass, f1index=0, f2index=1):
    nObs, nFea = features.shape
    colors = ['b', 'g', 'r']
    for i in range(nObs):
        plt.loglog(features[i, f1index], features[i, f2index], 'o' + colors[int(oClass[i])])

    plt.show()
    waitforEnter()


## -- 5 -- ##
def extratctSilence(data, threshold=256):
    if (data[0] <= threshold):
        s = [1]
    else:
        s = []
    for i in range(1, len(data)):
        if (data[i - 1] > threshold and data[i] <= threshold):
            s.append(1)
        elif (data[i - 1] <= threshold and data[i] <= threshold):
            s[-1] += 1

    return (s)


def extractFeaturesSilence(data, Class=0):
    features = []
    nObs, nSamp, nCols = data.shape
    oClass = np.ones((nObs, 1)) * Class
    for i in range(nObs):
        silence_features = np.array([])
        for c in range(nCols):
            silence = extratctSilence(data[i, :, c], threshold=0)
            if len(silence) > 0:
                silence_features = np.append(silence_features, [np.mean(silence), np.var(silence)])
            else:
                silence_features = np.append(silence_features, [0, 0])

        features.append(silence_features)

    return np.array(features), oClass


## -- 7 -- ##

def extractFeaturesWavelet(data, scales=[2, 4, 8, 16, 32], Class=0):
    features = []
    nObs, nSamp, nCols = data.shape
    oClass = np.ones((nObs, 1)) * Class
    for i in range(nObs):
        scalo_features = np.array([])
        for c in range(nCols):
            # fixed scales->fscales
            scalo, fscales = scalogram.scalogramCWT(data[i, :, c], scales)
            scalo_features = np.append(scalo_features, scalo)

        features.append(scalo_features)

    return np.array(features), oClass


## -- 11 -- ##
def distance(c, p):
    return (np.sqrt(np.sum(np.square(p - c))))


########### Main Code #############
Classes = {0: bcolors.OKGREEN + 'Normal' + bcolors.ENDC, 1: bcolors.OKGREEN + 'Normal2' + bcolors.ENDC, 2: bcolors.WARNING + 'Attack' + bcolors.ENDC}
plt.ion()
nfig = 1

## -- 1 -- ##
normal_use = np.loadtxt('normal_use.dat')
normal_use2 = np.loadtxt('normal_use.dat')
attack = np.loadtxt('attack.dat')

plt.figure(1)
plot3Classes(normal_use, 'Normal', normal_use2, 'Normal2', attack, 'Attack')

## -- 2 -- ##
normal_use_train, normal_use_test = breakTrainTest(normal_use, 100, 0.5)
normal_use_train2, normal_use_test2 = breakTrainTest(normal_use2, 100, 0.5)
attack_train, attack_test = breakTrainTest(attack, 56, 0.7)

plt.figure(2)
plt.subplot(3, 1, 1)
for i in range(10):
    plt.plot(normal_use_train[i, :, 0], 'b')
    plt.plot(normal_use_train[i, :, 1], 'g')
plt.title('Normal Use')
plt.ylabel('Bytes/sec')
plt.subplot(3, 1, 2)
for i in range(10):
    plt.plot(normal_use_train2[i, :, 0], 'b')
    plt.plot(normal_use_train2[i, :, 1], 'g')
plt.title('Normal Use 2')
plt.ylabel('Bytes/sec')
plt.subplot(3, 1, 3)
for i in range(10):
    plt.plot(attack_train[i, :, 0], 'b')
    plt.plot(attack_train[i, :, 1], 'g')
plt.title('Attack')
plt.ylabel('Bytes/sec')
plt.show()
waitforEnter()
## -- 3 -- ##
features_normal_use, oClass_normal_use = extractFeatures(normal_use_train, Class=0)
features_normal_use2, oClass_normal_use2 = extractFeatures(normal_use_train2, Class=1)
features_attack, oClass_attack = extractFeatures(attack_train, Class=2)

features = np.vstack((features_normal_use, features_normal_use2, features_attack))
oClass = np.vstack((oClass_normal_use, oClass_normal_use2, oClass_attack))

print('Train Stats Features Size:', features.shape)

## -- 4 -- ##
plt.figure(4)
plotFeatures(features, oClass, 0, 1)  # 0,8

## -- 5 -- ##
features_normal_useS, oClass_normal_use = extractFeaturesSilence(normal_use_train, Class=0)
features_normal_use2S, oClass_normal_use2 = extractFeaturesSilence(normal_use_train2, Class=1)
features_attackS, oClass_attack = extractFeaturesSilence(attack_train, Class=2)

featuresS = np.vstack((features_normal_useS, features_normal_use2S, features_attackS))
oClass = np.vstack((oClass_normal_use, oClass_normal_use2, oClass_attack))

print('Train Silence Features Size:', featuresS.shape)
plt.figure(5)
plotFeatures(featuresS, oClass, 0, 2)

## -- 6 -- ##
import scalogram

# mudar consoante janela de observacao
scales = range(2, 128)
plt.figure(6)

i = 0
data = normal_use_train[i, :, 1]
S, scalesF = scalogram.scalogramCWT(data, scales)
plt.plot(scalesF, S, 'b')

nObs, nSamp, nCol = normal_use_train2.shape
data = normal_use_train2[i, :, 1]
S, scalesF = scalogram.scalogramCWT(data, scales)
plt.plot(scalesF, S, 'g')

nObs, nSamp, nCol = attack_train.shape
data = attack_train[i, :, 1]
S, scalesF = scalogram.scalogramCWT(data, scales)
plt.plot(scalesF, S, 'r')

plt.show()
waitforEnter()

## -- 7 -- ##
# scales = [2, 4, 8, 16, 32, 64, 128, 256] # o 256 nao faz sentido para o tamanho da janela de observacao
scales = [2, 4, 8, 16, 32, 64]
features_normal_useW, oClass_normal_use = extractFeaturesWavelet(normal_use_train, scales, Class=0)
features_normal_use2W, oClass_normal_use2 = extractFeaturesWavelet(normal_use_train2, scales, Class=1)
features_attackW, oClass_attack = extractFeaturesWavelet(attack_train, scales, Class=2)

featuresW = np.vstack((features_normal_useW, features_normal_use2W, features_attackW))
oClass = np.vstack((oClass_normal_use, oClass_normal_use2, oClass_attack))

print('Train Wavelet Features Size:', featuresW.shape)
plt.figure(7)
plotFeatures(featuresW, oClass, 3, 9)

## -- 8 -- ##
#:1 para detecao
trainFeatures_normal_use, oClass_yt = extractFeatures(normal_use_train, Class=0)
trainFeatures_normal_use2, oClass_normal_use2 = extractFeatures(normal_use_train2, Class=1)
trainFeatures = np.vstack((trainFeatures_normal_use, trainFeatures_normal_use2))

trainFeatures_normal_useS, oClass_normal_use = extractFeaturesSilence(normal_use_train, Class=0)
trainFeatures_normal_use2S, oClass_normal_use2 = extractFeaturesSilence(normal_use_train2, Class=1)
trainFeaturesS = np.vstack((trainFeatures_normal_useS, trainFeatures_normal_use2S))

trainFeatures_normal_useW, oClass_normal_use = extractFeaturesWavelet(normal_use_train, scales, Class=0)
trainFeatures_normal_use2W, oClass_normal_use2 = extractFeaturesWavelet(normal_use_train2, scales, Class=1)
trainFeaturesW = np.vstack((trainFeatures_normal_useW, trainFeatures_normal_use2W))

o2trainClass = np.vstack((oClass_normal_use, oClass_normal_use2))
i2trainFeatures = np.hstack((trainFeatures, trainFeaturesS, trainFeaturesW))

#:2 para classificacao
# trainFeatures_normal_use, oClass_normal_use = extractFeatures(normal_use_train, Class=0)
# trainFeatures_normal_use2, oClass_normal_use2 = extractFeatures(normal_use_train2, Class=1)
# trainFeatures_attack, oClass_attack = extractFeatures(attack_train, Class=2)
# trainFeatures = np.vstack((trainFeatures_normal_use, trainFeatures_normal_use2, trainFeatures_attack))
#
# trainFeatures_normal_useS, oClass_normal_use = extractFeaturesSilence(normal_use_train, Class=0)
# trainFeatures_normal_use2S, oClass_normal_use2 = extractFeaturesSilence(normal_use_train2, Class=1)
# trainFeatures_attackS, oClass_attack = extractFeaturesSilence(attack_train, Class=2)
# trainFeaturesS = np.vstack((trainFeatures_normal_useS, trainFeatures_normal_use2S, trainFeatures_attackS))
#
# trainFeatures_normal_useW, oClass_normal_use = extractFeaturesWavelet(normal_use_train, scales, Class=0)
# trainFeatures_normal_use2W, oClass_normal_use2 = extractFeaturesWavelet(normal_use_train2, scales, Class=1)
# trainFeatures_attackW, oClass_attack = extractFeaturesWavelet(attack_train, scales, Class=2)
# trainFeaturesW = np.vstack((trainFeatures_normal_useW, trainFeatures_normal_use2W, trainFeatures_attackW))

# o3trainClass = np.vstack((oClass_normal_use, oClass_normal_use2, oClass_attack))
# i3trainFeatures = np.hstack((trainFeatures, trainFeaturesS, trainFeaturesW))

#:3 testar
testFeatures_normal_use, oClass_normal_use = extractFeatures(normal_use_test, Class=0)
testFeatures_normal_use2, oClass_normal_use2 = extractFeatures(normal_use_test2, Class=1)
testFeatures_attack, oClass_attack = extractFeatures(attack_test, Class=2)
testFeatures = np.vstack((testFeatures_normal_use, testFeatures_normal_use2, testFeatures_attack))

testFeatures_normal_useS, oClass_normal_use = extractFeaturesSilence(normal_use_test, Class=0)
testFeatures_normal_use2S, oClass_normal_use2 = extractFeaturesSilence(normal_use_test2, Class=1)
testFeatures_attackS, oClass_attack = extractFeaturesSilence(attack_test, Class=2)
testFeaturesS = np.vstack((testFeatures_normal_useS, testFeatures_normal_use2S, testFeatures_attackS))

testFeatures_normal_useW, oClass_normal_use = extractFeaturesWavelet(normal_use_test, scales, Class=0)
testFeatures_normal_use2W, oClass_normal_use2 = extractFeaturesWavelet(normal_use_test2, scales, Class=1)
testFeatures_attackW, oClass_attack = extractFeaturesWavelet(attack_test, scales, Class=2)
testFeaturesW = np.vstack((testFeatures_normal_useW, testFeatures_normal_use2W, testFeatures_attackW))

o3testClass = np.vstack((oClass_normal_use, oClass_normal_use2, oClass_attack))
i3testFeatures = np.hstack((testFeatures, testFeaturesS, testFeaturesW))

## -- 9 -- ##
from sklearn.preprocessing import MaxAbsScaler

i2trainScaler = MaxAbsScaler().fit(i2trainFeatures)
i2trainFeaturesN = i2trainScaler.transform(i2trainFeatures)

# i3trainScaler = MaxAbsScaler().fit(i3trainFeatures)
# i3trainFeaturesN = i3trainScaler.transform(i3trainFeatures)

i3AtestFeaturesN = i2trainScaler.transform(i3testFeatures)
# i3CtestFeaturesN = i3trainScaler.transform(i3testFeatures)

# Transformacao de NaN em 0
i2trainFeaturesN = np.nan_to_num(i2trainFeaturesN)
# i3trainFeaturesN = np.nan_to_num(i3trainFeaturesN)
i3AtestFeaturesN = np.nan_to_num(i3AtestFeaturesN)
# i3CtestFeaturesN = np.nan_to_num(i3CtestFeaturesN)

print(np.mean(i2trainFeaturesN, axis=0))
print(np.std(i2trainFeaturesN, axis=0))

## -- 10 -- ##
from sklearn.decomposition import PCA

pca = PCA(n_components=3, svd_solver='full')

i2trainPCA = pca.fit(i2trainFeaturesN)
i2trainFeaturesNPCA = i2trainPCA.transform(i2trainFeaturesN)

# i3trainPCA = pca.fit(i3trainFeaturesN)
# i3trainFeaturesNPCA = i3trainPCA.transform(i3trainFeaturesN)

i3AtestFeaturesNPCA = i2trainPCA.transform(i3AtestFeaturesN)
# i3CtestFeaturesNPCA = i3trainPCA.transform(i3CtestFeaturesN)

plt.figure(8)
plotFeatures(i2trainFeaturesNPCA, o2trainClass, 0, 1)

## -- 11 -- ##
from sklearn.preprocessing import MaxAbsScaler

# centroids = {}
# for c in range(2):  # Only the first two classes
#     pClass = (o2trainClass == c).flatten()
#     centroids.update({c: np.mean(i2trainFeaturesN[pClass, :], axis=0)})
# print('All Features Centroids:\n', centroids)
#
# AnomalyThreshold = 1.2
# print('\n-- Anomaly Detection based on Centroids Distances --')
# nObsTest, nFea = i3AtestFeaturesN.shape
# for i in range(nObsTest):
#     x = i3AtestFeaturesN[i]
#     dists = [distance(x, centroids[0]), distance(x, centroids[1])]
#     if min(dists) > AnomalyThreshold:
#         result = "Anomaly"
#     else:
#         result = "OK"
#
#     print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(i, Classes[
#         o3testClass[i][0]], *dists, result))
#
# ## -- 12 -- ##
# centroids = {}
# for c in range(2):  # Only the first two classes
#     pClass = (o2trainClass == c).flatten()
#     centroids.update({c: np.mean(i2trainFeaturesNPCA[pClass, :], axis=0)})
# print('All Features Centroids:\n', centroids)
#
# AnomalyThreshold = 1.2
# print('\n-- Anomaly Detection based on Centroids Distances (PCA Features) --')
# nObsTest, nFea = i3AtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     x = i3AtestFeaturesNPCA[i]
#     dists = [distance(x, centroids[0]), distance(x, centroids[1])]
#     if min(dists) > AnomalyThreshold:
#         result = "Anomaly"
#     else:
#         result = "OK"
#
#     print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(i, Classes[
#         o3testClass[i][0]], *dists, result))
#
# ## -- 13 -- ##
# from scipy.stats import multivariate_normal
#
# print('\n-- Anomaly Detection based Multivariate PDF (PCA Features) --')
# means = {}
# for c in range(2):
#     pClass = (o2trainClass == c).flatten()
#     means.update({c: np.mean(i2trainFeaturesNPCA[pClass, :], axis=0)})
# # print(means)
#
# covs = {}
# for c in range(2):
#     pClass = (o2trainClass == c).flatten()
#     covs.update({c: np.cov(i2trainFeaturesNPCA[pClass, :], rowvar=0)})
# # print(covs)
#
# AnomalyThreshold = 0.05
# nObsTest, nFea = i3AtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     x = i3AtestFeaturesNPCA[i, :]
#     probs = np.array([multivariate_normal.pdf(x, means[0], covs[0]), multivariate_normal.pdf(x, means[1], covs[1])])
#     if max(probs) < AnomalyThreshold:
#         result = "Anomaly"
#     else:
#         result = "OK"
#
#     print('Obs: {:2} ({}): Probabilities: [{:.4e},{:.4e}] -> Result -> {}'.format(i, Classes[o3testClass[i][0]], *probs,
#                                                                                   result))

## -- 14 -- ##
from sklearn import svm

print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
ocsvm = svm.OneClassSVM(gamma='scale', kernel='linear').fit(i2trainFeaturesNPCA)
rbf_ocsvm = svm.OneClassSVM(gamma='scale', kernel='rbf').fit(i2trainFeaturesNPCA)
poly_ocsvm = svm.OneClassSVM(gamma='scale', kernel='poly', degree=2).fit(i2trainFeaturesNPCA)

L1 = ocsvm.predict(i3AtestFeaturesNPCA)
L2 = rbf_ocsvm.predict(i3AtestFeaturesNPCA)
L3 = poly_ocsvm.predict(i3AtestFeaturesNPCA)

AnomResults = {-1: "Anomaly", 1: "OK"}

nObsTest, nFea = i3AtestFeaturesNPCA.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i, Classes[
        o3testClass[i][0]], AnomResults[L1[i]], AnomResults[L2[i]], AnomResults[L3[i]]))

## -- 15 -- ##
from sklearn import svm

print('\n-- Anomaly Detection based on One Class Support Vector Machines --')
ocsvm = svm.OneClassSVM(gamma='scale', kernel='linear').fit(i2trainFeaturesN)
rbf_ocsvm = svm.OneClassSVM(gamma='scale', kernel='rbf').fit(i2trainFeaturesN)
poly_ocsvm = svm.OneClassSVM(gamma='scale', kernel='poly', degree=2).fit(i2trainFeaturesN)

L1 = ocsvm.predict(i3AtestFeaturesN)
L2 = rbf_ocsvm.predict(i3AtestFeaturesN)
L3 = poly_ocsvm.predict(i3AtestFeaturesN)

AnomResults = {-1: "Anomaly", 1: "OK"}

nObsTest, nFea = i3AtestFeaturesN.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i, Classes[
        o3testClass[i][0]], AnomResults[L1[i]], AnomResults[L2[i]], AnomResults[L3[i]]))

## -- 16 -- ##
# centroids = {}
# for c in range(3):  # All 3 classes
#     pClass = (o3trainClass == c).flatten()
#     centroids.update({c: np.mean(i3trainFeaturesNPCA[pClass, :], axis=0)})
# print('PCA Features Centroids:\n', centroids)
#
# print('\n-- Classification based on Centroids Distances (PCA Features) --')
# nObsTest, nFea = i3CtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     x = i3CtestFeaturesNPCA[i]
#     dists = [distance(x, centroids[0]), distance(x, centroids[1]), distance(x, centroids[2])]
#     ndists = dists / np.sum(dists)
#     testClass = np.argsort(dists)[0]
#
#     print(
#         'Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f},{:.4f}] -> Classification: {} -> {}'.format(
#             i, Classes[o3testClass[i][0]], *ndists, testClass, Classes[testClass]))
#
# ## -- 17-- #
# from scipy.stats import multivariate_normal
#
# print('\n-- Classification based on Multivariate PDF (PCA Features) --')
# means = {}
# for c in range(3):
#     pClass = (o3trainClass == c).flatten()
#     means.update({c: np.mean(i3trainFeaturesNPCA[pClass, :], axis=0)})
# # print(means)
#
# covs = {}
# for c in range(3):
#     pClass = (o3trainClass == c).flatten()
#     covs.update({c: np.cov(i3trainFeaturesNPCA[pClass, :], rowvar=0)})
# # print(covs)
#
# nObsTest, nFea = i3CtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     x = i3CtestFeaturesNPCA[i, :]
#     probs = np.array([multivariate_normal.pdf(x, means[0], covs[0]), multivariate_normal.pdf(x, means[1], covs[1]),
#                       multivariate_normal.pdf(x, means[2], covs[2])])
#     testClass = np.argsort(probs)[-1]
#
#     print('Obs: {:2} ({}): Probabilities: [{:.4e},{:.4e},{:.4e}] -> Classification: {} -> {}'.format(i, Classes[
#         o3testClass[i][0]], *probs, testClass, Classes[testClass]))

## -- 18 -- #
# print('\n-- Classification based on Support Vector Machines --')
# svc = svm.SVC(kernel='linear').fit(i3trainFeaturesN, o3trainClass)
# rbf_svc = svm.SVC(kernel='rbf').fit(i3trainFeaturesN, o3trainClass)
# poly_svc = svm.SVC(kernel='poly', degree=2).fit(i3trainFeaturesN, o3trainClass)
#
# L1 = svc.predict(i3CtestFeaturesN)
# L2 = rbf_svc.predict(i3CtestFeaturesN)
# L3 = poly_svc.predict(i3CtestFeaturesN)
# print('\n')
#
# nObsTest, nFea = i3CtestFeaturesN.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i, Classes[
#         o3testClass[i][0]], Classes[L1[i]], Classes[L2[i]], Classes[L3[i]]))
#
# ## -- 19 -- #
# print('\n-- Classification based on Support Vector Machines  (PCA Features) --')
# svc = svm.SVC(kernel='linear').fit(i3trainFeaturesNPCA, o3trainClass)
# rbf_svc = svm.SVC(kernel='rbf').fit(i3trainFeaturesNPCA, o3trainClass)
# poly_svc = svm.SVC(kernel='poly', degree=2).fit(i3trainFeaturesNPCA, o3trainClass)
#
# L1 = svc.predict(i3CtestFeaturesNPCA)
# L2 = rbf_svc.predict(i3CtestFeaturesNPCA)
# L3 = poly_svc.predict(i3CtestFeaturesNPCA)
# print('\n')
#
# nObsTest, nFea = i3CtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i, Classes[
#         o3testClass[i][0]], Classes[L1[i]], Classes[L2[i]], Classes[L3[i]]))
#
# ## -- 20a -- ##
# from sklearn.neural_network import MLPClassifier
#
# print('\n-- Classification based on Neural Networks --')
#
# alpha = 1
# max_iter = 100000
# clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(20,), max_iter=max_iter)
# clf.fit(i3trainFeaturesN, o3trainClass)
# LT = clf.predict(i3CtestFeaturesN)
#
# nObsTest, nFea = i3CtestFeaturesN.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Classification->{}'.format(i, Classes[o3testClass[i][0]], Classes[LT[i]]))
#
# ## -- 20b -- ##
# from sklearn.neural_network import MLPClassifier
#
# print('\n-- Classification based on Neural Networks (PCA Features) --')
#
# alpha = 1
# max_iter = 100000
# clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(20,), max_iter=max_iter)
# clf.fit(i3trainFeaturesNPCA, o3trainClass)
# LT = clf.predict(i3CtestFeaturesNPCA)
#
# nObsTest, nFea = i3CtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Classification->{}'.format(i, Classes[o3testClass[i][0]], Classes[LT[i]]))
