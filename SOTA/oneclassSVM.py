from LoadDownstreamData import GetTrainSet,GetValSet,GetTestSet
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve,auc,accuracy_score,precision_recall_curve
import pickle,os

path0 =os.path.abspath('..')
path = os.getcwd()

train_x, train_y = GetTrainSet(path0)
test_x, test_y  = GetTestSet(path0)
test_y.tolist()
epoch = 10

clf = OneClassSVM(gamma='auto')

for i in range(epoch):
    y_pred = clf.fit_predict(test_x)
    for j in range(len(y_pred)):
        if y_pred[j] == -1:
            y_pred[j] = 1
        else:
            y_pred[j] = 0
    acc = accuracy_score(test_y,y_pred)
    fpr, tpr, thersholds = roc_curve(test_y,y_pred)
    roc_auc = auc(fpr, tpr)
    prec, recall, thersholds2 = precision_recall_curve(test_y,y_pred)
    prc_auc = auc(recall, prec)
    print(y_pred)
    print('acc: ',acc)
    print('roc_auc: ',roc_auc)
    print('prc_auc: ',prc_auc)

'''
[0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
acc:  0.5123152709359606
roc_auc:  0.495578231292517
prc_auc:  0.7529022988505748'''