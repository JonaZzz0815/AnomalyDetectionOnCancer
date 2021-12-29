from LoadDownstreamData import path
from LoadDownstreamData import train_x,train_y
from LoadDownstreamData import val_x,val_y
from LoadDownstreamData import test_x,test_y
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score,roc_curve,auc
import pickle
epoch = 5
clf = KMeans(n_clusters=2,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=0.0001,
                random_state=None,
                copy_x=True,
                algorithm='auto'
                )

for i in range(epoch):
    clf.fit(train_x)
    y_train_pred = clf.labels_
    fpr, tpr, thersholds = roc_curve(train_y,y_train_pred)
    roc_auc = auc(fpr, tpr)
    print(y_train_pred)
    print(roc_auc)

#使用dump()将数据序列化到文件中
fw = open(path+'/SOTA/kmeans.txt','wb')
# Pickle dictionary using protocol 0.
pickle.dump(clf, fw)
fw.close()

y_val_pred = clf.predict(val_x)
fpr, tpr, thersholds = roc_curve(val_y,y_val_pred)
roc_auc_val = auc(fpr, tpr)
print(y_val_pred)
print(roc_auc_val)