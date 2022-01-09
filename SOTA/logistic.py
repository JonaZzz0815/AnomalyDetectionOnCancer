from LoadDownstreamData import GetTrainSet,GetValSet,GetTestSet
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import roc_curve,auc,accuracy_score,precision_recall_curve
import pickle,os

path0 =os.path.abspath('..')
path = os.getcwd()

train_x, train_y = GetTrainSet(path0)
test_x, test_y  = GetTestSet(path0)
train_y.tolist()
test_y.tolist()
epoch = 10

clf = LogisticRegression(solver='liblinear')

clf.fit(train_x,train_y)
# train_y = train_y[0]
# acc = accuracy_score(train_y,y_pred)
# fpr, tpr, thersholds = roc_curve(train_y,y_pred)
# roc_auc = auc(fpr, tpr)
# prec, recall, thersholds2 = precision_recall_curve(train_y,y_pred)
# prc_auc = auc(recall, prec)
# print(y_pred)
# print('acc on trainset: ',acc)
# print('roc_auc on trainset: ',roc_auc)
# print('prc_auc on trainset: ',prc_auc)

y_pred = clf.predict(test_x)
acc = accuracy_score(test_y,y_pred)
fpr, tpr, _ = roc_curve(test_y,y_pred)
roc_auc = auc(fpr, tpr)
prec, recall, _ = precision_recall_curve(test_y,y_pred)
prc_auc = auc(recall, prec)
print(y_pred)
print('acc: ',acc)
print('roc_auc: ',roc_auc)
print('prc_auc: ',prc_auc)

'''
setting solver = 'liblienar', which is good for small datasets:
[0. 1. 1. 0. 0. 0. 1. 1. 1. 0. 1. 0. 0. 1. 1. 1. 0. 1. 1. 0. 0. 1. 0. 0.
 1. 0. 1. 0. 0. 1. 1. 1. 1. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 1. 0. 0.
 1. 1. 1. 1. 0. 1. 0. 0. 0. 1. 0. 1. 1. 1. 1. 1. 0. 0. 0. 0. 1. 1. 1. 1.
 0. 1. 0. 0. 1. 1. 0. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 0. 0. 1. 0. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 0. 1. 0. 0. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 0.
 0. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 1. 1. 1. 1. 0. 1. 1. 1.
 0. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 0. 1. 1. 1. 1. 0. 1. 1.]
acc:  0.6059113300492611
roc_auc:  0.6003401360544217
prc_auc:  0.7388250319284801'''
