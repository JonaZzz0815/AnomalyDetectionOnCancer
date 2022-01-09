from LoadDownstreamData import GetTrainSet,GetValSet,GetTestSet
from pyod.models.iforest import IForest
from sklearn.metrics import roc_curve,auc,accuracy_score,precision_recall_curve
import pickle,os
path0 =os.path.abspath('..')
path = os.getcwd()

train_x, train_y = GetTrainSet(path0)
test_x, test_y  = GetTestSet(path0)
test_y.tolist()
epoch = 10
clf_name = 'iforest'
clf = IForest(contamination=0.4)

for i in range(epoch):
    y_pred = clf.fit_predict(test_x)
    acc = accuracy_score(test_y, y_pred)
    fpr, tpr, thersholds = roc_curve(test_y, y_pred)
    roc_auc = auc(fpr, tpr)
    prec, recall, thersholds2 = precision_recall_curve(test_y, y_pred)
    prc_auc = auc(recall, prec)
    print(y_pred)
    print('acc: ', acc)
    print('roc_auc: ', roc_auc)
    print('prc_auc: ', prc_auc)

#使用dump()将数据序列化到文件中
fw = open(path+'/IForest.txt','wb')
# Pickle dictionary using protocol 0.
pickle.dump(clf, fw)
fw.close()

'''
[0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0
 1 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 1 0 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0
 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
acc:  0.46798029556650245
roc_auc:  0.48163265306122444
prc_auc:  0.49359605911330046

setting contamination = 0.4:
[0 1 1 0 1 0 0 0 1 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1
 1 0 0 0 0 1 1 1 1 1 1 1 1 1 0 1 1 0 0 0 0 0 1 1 0 1 0 0 1 1 1 1 1 0 0 0 0
 0 1 0 0 1 1 1 1 0 0 0 0 0 0 1 0 0 1 0 1 0 0 1 1 0 0 0 0 0 1 0 1 1 1 0 0 0
 0 0 0 0 0 1 0 1 0 1 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 1 1 1 0 1 0 0 0 0 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 0 0 0 0
 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 1 1]
acc:  0.458128078817734
roc_auc:  0.4615646258503402
prc_auc:  0.5805449127288207'''
