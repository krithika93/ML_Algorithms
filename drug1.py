import os
import itertools
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
with open('cs584_hw2.txt', 'r') as f:
	data1 = f.readlines()
z_o_val =[]
z_o_row = []
z_o_col = []
class_labels=[]
for (rindex,row) in enumerate(data1): 
	data2 = row.split("\t")
	class_labels= class_labels + list(data2[0])
	non_zero_indices = data2[1].replace("\n","").split(" ")
	non_zero_indices = [int(x) for x in non_zero_indices if x != ""]
	ones = [(1,x) for x in  non_zero_indices]
	z_o = ones
	z_o.sort(key=lambda x: x[1])
	print(len(ones))
	z_o_val = z_o_val +[x[0] for x in z_o] 
	z_o_col = z_o_col+[x[1] for x in z_o]
	z_o_row = z_o_row + [rindex for x in z_o]
	

data = np.array(z_o_val)
print(len(data))
row = np.array(z_o_row)
print(len(row))
col = np.array(z_o_col)
print(len(col))
c_matrix = csr_matrix((data,(row,col)),shape=(800,100001)).toarray()
print(c_matrix)
print(c_matrix.shape)

counter = Counter()


for index  in range(800):
  
    label = class_labels[index]
    counter[label] += 1
print(counter.items() )
print(len(class_labels))
labels = np.array(class_labels)
X_train, X_test, y_train, y_test = train_test_split(
    c_matrix,
    labels,
    test_size=0.3,
    random_state=0
)
with open('cs584_hw2_test.txt', 'r') as f:
	data1 = f.readlines()
z_o_val =[]
z_o_row = []
z_o_col = []
class_labels=[]
for (rindex,row) in enumerate(data1): 
	non_zero_indices = row.replace("\n","").split(" ")
	non_zero_indices = [int(x) for x in non_zero_indices if x != ""]

	ones = [(1,x) for x in  non_zero_indices]
	z_o = ones
	z_o.sort(key=lambda x: x[1])
	print(len(ones))
	z_o_val = z_o_val +[x[0] for x in z_o] 
	z_o_col = z_o_col+[x[1] for x in z_o]
	z_o_row = z_o_row + [rindex for x in z_o]


data = np.array(z_o_val)
print(len(data))
row = np.array(z_o_row)
print(len(row))
col = np.array(z_o_col)
print(len(col))
c_matrix1 = csr_matrix((data,(row,col)),shape=(350,100001)).toarray()
c_matrix2 =c_matrix1
c_matrix3 =c_matrix1

print(c_matrix1)
print(c_matrix1.shape)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
X_test, y_test = SMOTE().fit_resample(X_test, y_test)
counter = Counter()
for index  in range(800):
  
    label = y_train[index]
    counter[label] += 1
print(counter.items() )
clf = DecisionTreeClassifier(criterion = "entropy",class_weight = "balanced",splitter = "random")
clf.fit(X_train,y_train)
print(clf.get_params(deep=True))
clf.score(X=X_test, y=y_test) # 1.0
y_pred = clf.predict(X_test)
print(len(y_pred),"ypred")
print(len(y_test),"ytest")
scores = cross_val_score(clf, X_test, y_test, cv=10)
print(scores)
f1 = f1_score(y_test,y_pred,average=None)
print(f1)
# print("KNeighbors")
# neigh = KNeighborsClassifier(n_neighbors=1,algorithm="brute")
# neigh.fit(X_train, y_train)
# y_pred1= neigh.predict(X_test)
# f1 = f1_score(y_test,y_pred1,average=None)
# print(f1)
print("Support Vector Machine")
sv = LinearSVC(C=0.01, penalty='l2', dual=True).fit(X_train,y_train)
modelsv = SelectFromModel(sv, prefit=True)
X_new = modelsv.transform(X_train)
X_new1 = modelsv.transform(X_test)
c_matrix1 = modelsv.transform(c_matrix1)
sv.fit(X_new, y_train)
y_pred1 = sv.predict(X_new1)
f1 = f1_score(y_test,y_pred1,average=None)
print(f1)
print("LogisticRegression")
lg= LogisticRegression(C=0.01,random_state=0, class_weight = "balanced", solver='liblinear',
                         multi_class='ovr').fit(X_train,y_train)
modellg = SelectFromModel(lg, prefit=True)

X_new = modellg.transform(X_train)
X_new1 = modellg.transform(X_test)
c_matrix2 = modellg.transform(c_matrix2)

lg.fit(X_new, y_train)
y_pred1 = lg.predict(X_new1)
f1 = f1_score(y_test,y_pred1,average=None)
print(f1)
print("RandomForests")
rf = RandomForestClassifier(criterion = "entropy",n_estimators=10, max_depth=10,
      min_samples_split=2, random_state=0)
rf.fit(X_train,y_train)   
y_pred1 = rf.predict(X_test)
f1 = f1_score(y_test,y_pred1,average=None)
print(f1)
y_pred2 = clf.predict(c_matrix3)
print("Support Vector Machine")
y_pred3 = sv.predict(c_matrix1)
y_pred4 = lg.predict(c_matrix2)
y_pred5 = rf.predict(c_matrix3)
y_pred6= neigh.predict(c_matrix3)
with open('sv.txt', 'w') as f:
    for item in y_pred3:
        f.write("%s\n" % item)
		
with open('clf.txt', 'w') as f:
    for item in y_pred2:
        f.write("%s\n" % item)
with open('lg.txt', 'w') as f:
	for item in y_pred4:
		f.write("%s\n" % item)
with open('rf.txt', 'w') as f:
	for item in y_pred5:
		f.write("%s\n" % item)		

# with open('knn.txt', 'w') as f:
	# for item in y_pred6:
		# f.write("%s\n" % item)		