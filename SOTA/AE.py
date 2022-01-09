from LoadDownstreamData import GetTrainSet,GetValSet,GetTestSet
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Input
from keras.models import Model
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve,auc,accuracy_score,precision_recall_curve
import pickle,os

path0 =os.path.abspath('..')
path = os.getcwd()

train_x, train_y = GetTrainSet(path0)
test_x, test_y  = GetTestSet(path0)
val_x, val_y = GetValSet(path0)
semi_train_x = train_x[:191,:]
semi_train_y = train_y[:191,:]
semi_val_x = val_x[:60,:]
semi_val_y = val_y[:60,:]
semi_train_y.tolist()
semi_val_y.tolist()
train_y.tolist()
test_y.tolist()
val_y.tolist()
# print(semi_train_x.shape)
# print('val',semi_val_y)
# print('train',semi_train_y)

# 设置Autoencoder的参数
# 隐藏层节点数分别为16，8，8，16
# epoch为50，batch size为32
input_dim = semi_train_x.shape[1]
encoding_dim = 16
num_epoch = 10
batch_size = 32

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['mse'])

checkpointer = ModelCheckpoint(filepath="AE",
                               verbose=0,
                               save_best_only=True)

history = autoencoder.fit(semi_train_x, semi_train_x,
                          epochs=num_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(semi_val_x,semi_val_x),
                          verbose=1,
                          callbacks=[checkpointer]).history
print(history)
test_pred = autoencoder.predict(test_x)
mse_test = np.mean(np.power(test_x - test_pred, 2), axis=1)
print('mse: ',mse_test)
threshold = history['mse'][num_epoch-1]
print(threshold)
y_pred = np.empty(203,)
for i in range(len(mse_test)):
    epsilon = random.uniform(-0.02*threshold,0.02*threshold)
    alpha = threshold + epsilon
    if mse_test[i] > alpha:
        y_pred[i] = (1)
    else:
        y_pred[i] = (0)

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
setting epoch = 10 and epsilon = 2%:
[1. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.
 1. 1. 0. 1. 0. 0. 0. 1. 0. 1. 1. 1. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 1. 1.
 0. 1. 1. 1. 0. 0. 0. 1. 1. 0. 0. 0. 1. 1. 1. 0. 0. 0. 1. 1. 1. 0. 0. 1.
 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 1. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0.
 0. 0. 1. 1. 0. 1. 1. 0. 0. 0. 1. 1. 1. 0. 0. 1. 0. 1. 0. 0. 0. 0. 1. 1.
 1. 0. 1. 1. 1. 0. 0. 1. 1. 1. 0. 1. 1. 1. 0. 1. 0. 1. 1. 1. 1. 0. 1. 0.
 1. 0. 0. 0. 0. 0. 1. 1. 1. 1. 0. 0. 1. 0. 1. 1. 1. 0. 0. 0. 0. 1. 0. 0.
 1. 0. 0. 0. 1. 0. 1. 1. 0. 1. 0. 0. 0. 1. 1. 0. 1. 1. 1. 0. 1. 0. 1. 0.
 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
acc:  0.5615763546798029
roc_auc:  0.5642857142857143
prc_auc:  0.672373696872494
'''