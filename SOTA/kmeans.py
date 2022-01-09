from LoadDownstreamData import GetTrainSet,GetValSet,GetTestSet
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve,auc,accuracy_score,precision_recall_curve
import pickle,os

path0 =os.path.abspath('..')
path = os.getcwd()

train_x, train_y = GetTrainSet(path0)
test_x, test_y  = GetTestSet(path0)
test_y.tolist()
epoch = 10
clf = KMeans(n_clusters=2,
                init='k-means++',
                n_init=10,
                max_iter=1000,
                tol=0.0001,
                random_state=None,
                copy_x=True,
                algorithm='auto'
                )

for i in range(epoch):
    y_pred = clf.fit_predict(test_x)
    print(y_pred.shape)
    acc = accuracy_score(test_y,y_pred)
    fpr, tpr, thersholds = roc_curve(test_y,y_pred)
    roc_auc = auc(fpr, tpr)
    prec, recall, thersholds2 = precision_recall_curve(test_y,y_pred)
    prc_auc = auc(recall, prec)
    print(y_pred)
    print('acc: ',acc)
    print('roc_auc: ',roc_auc)
    print('prc_auc: ',prc_auc)

#使用dump()将数据序列化到文件中
fw = open(path+'/kmeans.txt','wb')
# Pickle dictionary using protocol 0.
pickle.dump(clf, fw)
fw.close()

'''
 k-means best results:
[0 0 0 0 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 0 0 0 0 0
 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0
 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 1 1 1 0 0 0 0 0
 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1
 1 1 0 1 0 0 1 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1]
acc:  0.5566502463054187
roc_auc:  0.5561224489795918
prc_auc:  0.6822660098522167
'''
