import os
import sys 
sys.path.append("..") 
from LoadDownstreamData import GetTestSet,GetValSet,GetTrainSet
from pyod.models.copod import COPOD
from sklearn.metrics import roc_auc_score,roc_curve,auc
import pickle
path = os.path.abspath(os.path.dirname(os.getcwd()))

train_x,train_y = GetTrainSet(path)
val_x,val_y = GetValSet(path)
test_x,test_y = GetTestSet(path)

epoch = 1
clf_name = 'copod'
clf = COPOD()

for i in range(epoch):
    clf.fit(train_x)
    y_train_pred = clf.labels_
    fpr, tpr, thersholds = roc_curve(train_y,y_train_pred)
    roc_auc = auc(fpr, tpr)
    print(y_train_pred)
    print(roc_auc)

#使用dump()将数据序列化到文件中
fw = open(path+'/SOTA/copod.txt','wb')
# Pickle dictionary using protocol 0.
pickle.dump(clf, fw)
fw.close()

y_val_pred = clf.predict(val_x)
fpr, tpr, thersholds = roc_curve(val_y,y_val_pred)
roc_auc_val = auc(fpr, tpr)
print(y_val_pred)
print(roc_auc_val)