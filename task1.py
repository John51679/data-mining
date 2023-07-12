import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split as ttp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ds = pd.read_csv("winequality-red.csv")

X = ds.iloc[:,0:11].values

Y = ds.iloc[:,11].values

Y = np.reshape(Y,[np.size(Y),1])

X_training, X_testing, Y_training, Y_testing = ttp(X,Y,test_size=0.25,random_state= 1234)

sc = StandardScaler()
X_training = sc.fit_transform(X_training)
X_testing = sc.fit_transform(X_testing)

clf = svm.SVC()
clf.fit(X_training,Y_training)

result = clf.predict(X_testing)

print(classification_report(Y_testing,result))
print(confusion_matrix(Y_testing,result))

"""Begin task B"""
X_training = np.append(X_training,Y_training,axis=1)

size = int(np.size(X_training[:,8]) * 0.33)
if float(size) != np.size(X_training[:,8]) * 0.33:
    size += 1

for i in range(size):
    X_training[i,8] = None

np.random.shuffle(X_training)

Y_training = X_training[:,11]
Y_training = Y_training.astype(int)
Y_training = np.reshape(Y_training,[np.size(Y_training),1])
X_training = np.delete(X_training,11,1)
"""Method 1"""

X_training_1 = np.delete(X_training,8,1)
X_testing_1 = np.delete(X_testing,8,1)

clf = svm.SVC()
clf.fit(X_training_1,Y_training)

result = clf.predict(X_testing_1)

print(classification_report(Y_testing,result))
print(confusion_matrix(Y_testing,result))

"""End of method 1"""

"""Method 2"""

X_training_2 = np.copy(X_training)
SUM = 0
AVG = 0.0

for i in range(np.size(X_training_2[:,8])):
    if not np.isnan(X_training_2[i,8]):
        SUM += X_training_2[i,8]

AVG = SUM / (np.size(X_training_2[:,8]) - size)

for i in range(np.size(X_training_2[:,8])):
    if np.isnan(X_training_2[i,8]):
        X_training_2[i,8] = AVG

clf = svm.SVC()
clf.fit(X_training_2,Y_training)

result = clf.predict(X_testing)

print(classification_report(Y_testing,result))
print(confusion_matrix(Y_testing,result))

"""End of method 2"""

"""Method 3"""

X_training_3 = np.copy(X_training)

ph_total = np.append(X_training_3,Y_training,axis=1)

ph_training_X = np.zeros([1,12])
ph_testing_X = np.zeros([1,12])

for i in range(np.size(ph_total[:,0])):
    if np.isnan(X_training_3[i,8]):
        ph_testing_X = np.append(ph_testing_X,[ph_total[i,:]],axis=0)
    else:
        ph_training_X = np.append(ph_training_X,[ph_total[i,:]],axis=0)

ph_training_X = np.delete(ph_training_X,0,0)
ph_testing_X = np.delete(ph_testing_X,0,0)

ph_testing_X = np.delete(ph_testing_X,8,1)

ph_training_Y = ph_training_X[:,8]
ph_training_X = np.delete(ph_training_X,8,1)

ph_training_Y = ph_training_Y.astype(int)
ph_training_Y = np.reshape(ph_training_Y,[np.size(ph_training_Y),1])
clf = LogisticRegression()
clf.fit(ph_training_X,ph_training_Y)

result = clf.predict(ph_testing_X)
counter = 0
for i in range(np.size(X_training_3[:,0])):
    if np.isnan(X_training_3[i,8]):
        X_training_3[i,8] = result[counter]
        counter += 1

clf = svm.SVC()
clf.fit(X_training_3,Y_training)
result = clf.predict(X_testing)
print(classification_report(Y_testing,result))
print(confusion_matrix(Y_testing,result))

"""End of method 3"""

"""Method 4"""

X_training_4 = np.copy(X_training)

X_training_4_not_ph = np.copy(X_training_4)
X_training_4_not_ph = np.delete(X_training_4_not_ph,8,1)

kmeans = KMeans(n_clusters=6)
kmeans.fit(X_training_4_not_ph)
result = kmeans.predict(X_training_4_not_ph)

cl3 = []
cl4 = []
cl5 = []
cl6 = []
cl7 = []
cl8 = []

for i in range(np.size(X_training_4_not_ph[:,0])):

    if result[i] == 0:
        cl3.append(X_training_4[i,8])
    elif result[i] == 1:
        cl4.append(X_training_4[i,8])
    elif result[i] == 2:
        cl5.append(X_training_4[i,8])
    elif result[i] == 3:
        cl6.append(X_training_4[i,8])
    elif result[i] == 4:
        cl7.append(X_training_4[i,8])
    elif result[i] == 5:
        cl8.append(X_training_4[i,8])
"""
SUM = [0,0,0,0,0,0]

SUM[0] = sum(cl3)
SUM[1] = sum(cl4)
SUM[2] = sum(cl5)
SUM[3] = sum(cl6)
SUM[4] = sum(cl7)
SUM[5] = sum(cl8)

AVG = [0.0,0.0,0.0,0.0,0.0,0.0]

AVG[0] = SUM[0] / len(cl3)
AVG[1] = SUM[1] / len(cl4)
AVG[2] = SUM[2] / len(cl5)
AVG[3] = SUM[3] / len(cl6)
AVG[4] = SUM[4] / len(cl7)
AVG[5] = SUM[5] / len(cl8)
"""

AVG = [np.nanmean(cl3),np.nanmean(cl4),np.nanmean(cl5),np.nanmean(cl6),np.nanmean(cl7),np.nanmean(cl8)]

for i in range(np.size(X_training_4[:,0])):

    if result[i] == 0 and np.isnan(X_training_4[i,8]):
        X_training_4[i,8] = AVG[0]
    elif result[i] == 1 and np.isnan(X_training_4[i,8]):
        X_training_4[i,8] = AVG[1]
    elif result[i] == 2 and np.isnan(X_training_4[i,8]):
        X_training_4[i,8] = AVG[2]
    elif result[i] == 3 and np.isnan(X_training_4[i,8]):
        X_training_4[i,8] = AVG[3]
    elif result[i] == 4 and np.isnan(X_training_4[i,8]):
        X_training_4[i,8] = AVG[4]
    elif result[i] == 5 and np.isnan(X_training_4[i,8]):
        X_training_4[i,8] = AVG[4]

clf = svm.SVC()
clf.fit(X_training_4,Y_training)
result = clf.predict(X_testing)
print(classification_report(Y_testing,result))
print(confusion_matrix(Y_testing,result))

"""End of method 4"""
