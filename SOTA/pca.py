from LoadDownstreamData import GetTrainSet,GetValSet,GetTestSet
from pyod.models.pca import PCA
from sklearn.metrics import roc_curve,auc,accuracy_score,precision_recall_curve
import pickle,os
path0 =os.path.abspath('..')
path = os.getcwd()

train_x, train_y = GetTrainSet(path0)
test_x, test_y  = GetTestSet(path0)
test_y.tolist()
epoch = 10
clf_name = 'PCA'
clf = PCA(contamination=0.4)

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
print(clf_name)

#使用dump()将数据序列化到文件中
fw = open(path+'/PCA.txt','wb')
# Pickle dictionary using protocol 0.
pickle.dump(clf, fw)
fw.close()
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 1
 1 0 0 0 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
acc:  0.43842364532019706
roc_auc:  0.4520408163265306
prc_auc:  0.4152709359605911

setting contamination = 0.4:
[0 1 1 0 1 0 1 1 1 1 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1
 1 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 0 1 0 0 0 1 0 1 0 0 0 0 1 1 1 0 0 0 1
 1 1 0 0 1 1 1 1 0 0 0 0 0 0 1 0 0 1 0 1 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 1 0
 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
 1 0 0 1 0 1 1 1 0 0 0 0 0 0 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 0
 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1]
acc:  0.4187192118226601
roc_auc:  0.42210884353741496
prc_auc:  0.5466581524052789
'''
