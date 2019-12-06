# -*- coding: utf-8 -*-
#encoding = utf-8
from numpy.random import seed
seed(2018)
import random
random.seed(2018)
from tensorflow import set_random_seed
set_random_seed((2018))
import os
import sys
import gc
import numpy as np
import scipy.io as sio
import sklearn.metrics
import keras
from collections import Counter
import time
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, LSTM, Conv3D, MaxPool3D, Conv1D, MaxPool1D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import optimizers
from keras import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA, IncrementalPCA
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt


raw_path = r'/data/kuang/hcp/tfMRI_motor/'
procData_path = os.path.join(raw_path, 'proc_hcp/stdProc_meanPSC')
subname_file = os.path.join(raw_path, 'subject_name995.txt')
subject_list = np.genfromtxt(subname_file, dtype=str).tolist()
condition_name = ['lf', 'lh', 'rf', 'rh', 't']
movement_rep = 2             #######################  Repeat times of each action
mat_file_name = 'z_sub_%s_motor'

DIMX = 91
DIMY = 109
DIMZ = 91
numClass = 5
num_use_subject = 995       #  200, 500, 995
num_mem_subject = 500       #  160, 400, 500
samplePerSubject = int(len(condition_name) * movement_rep)

def main():
    k = 5
    use_subname_file = subject_list[:num_use_subject]
    train_set, test_set, valid_set = randomSplit_TrainAndTest(use_subname_file, k)

    all_result_value= []

    #cross validation
    for i in range(k):
        print('##### Round ', i, ' -------------')
        train_name = train_set[i]
        test_name = test_set[i]
        valid_name = valid_set[i]

        result = cnn3d_method(train_name, test_name, valid_name, i)

        gc.collect()

        all_result_value.append(result[:-1])

    print('each : test_acc, num_hit, test_precision, test_recall, test_f1, durTime')
    print('cnn3d_method : ', all_result_value)

    print('ave : test_acc, num_hit, test_precision, test_recall, test_f1, durTime')
    print('cnn3d_method : ', np.mean(all_result_value, axis=0))
    print('cnn3d_method std: ', np.std(all_result_value, axis=0))

    print('Finished.')

# 3D CNN ----------------------------------------------
def cnn3d_method(train_name, test_name, valid_name, round_index):
    print('*** Run 3d CNN ...')

    model = Sequential()
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last',
                     input_shape=(DIMX, DIMY, DIMZ, 1), kernel_initializer=initializers.glorot_normal()))
    model.add(LeakyReLU())
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), kernel_initializer=initializers.glorot_normal()))
    model.add(LeakyReLU())
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(units=128, kernel_initializer=initializers.glorot_normal()))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=numClass, activation='softmax'))

    my_adam = optimizers.Adam(lr=0.0025, beta_1=0.9, beta_2=0.999, amsgrad=True)

    model.compile(loss='categorical_crossentropy', optimizer=my_adam, metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='loss', patience=6, mode='auto')
    # checkpoint
    bestModelSavePath = '/data/kuang/hcp_clj/para/cnn3d_%s_weights_best_995.hdf5' % str(round_index)
    checkpoint = ModelCheckpoint(bestModelSavePath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    lrate = LearningRateScheduler(sch_func, verbose=1)

    # load some train data to memory
    mem_train_name = train_name[:num_mem_subject]
    disk_train_name = set(train_name[num_mem_subject:])
    mem_train_x, mem_train_y = [], []
    for name in mem_train_name:
        datafile_name = os.path.join(procData_path, mat_file_name % name)
        datafile = sio.loadmat(datafile_name)
        for (j, cd) in enumerate(condition_name):
            for i_rep in range(movement_rep):
                sheetname = cd + '_' + str(i_rep)
                mem_train_x.append(datafile[sheetname])
                mem_train_y.append(j)
        del datafile
    gc.collect()
    print('mem train x size: ', sys.getsizeof(mem_train_x), len(mem_train_x), len(mem_train_name))
    # load valid data to memory
    valid_x, valid_y = [], []
    for name in valid_name:
        datafile_name = os.path.join(procData_path, mat_file_name % name)
        datafile = sio.loadmat(datafile_name)
        for (j, cd) in enumerate(condition_name):
            for i_rep in range(movement_rep):
                sheetname = cd + '_' + str(i_rep)
                valid_x.append(datafile[sheetname])
                valid_y.append(j)
        del datafile
    gc.collect()
    print('mem valid x size: ', sys.getsizeof(valid_x), len(valid_y), len(valid_name))

    # generator 3
    def generator_CNN3D_MandD(subject_name0, batch_size, train_valid_test_flag):
        # train_valid_test_flag: Train, Valid, Test, Info
        subject_name = np.array(subject_name0)
        if train_valid_test_flag in ['Train']:
            while True:
                np.random.shuffle(subject_name)
                mem_random_index = [xx for xx in range(samplePerSubject * num_mem_subject)]
                np.random.shuffle(mem_random_index)
                mem_visit_time, disk_visit_time = 0, 0
                batch_subject_size = int(batch_size / samplePerSubject)
                itr_time = int(len(subject_name) / batch_subject_size)
                for itr in range(itr_time):
                    subjectNameBatch = subject_name[itr * batch_subject_size: (itr + 1) * batch_subject_size]
                    data_x = []
                    data_y = []
                    for name in subjectNameBatch:
                        if name in disk_train_name:
                            datafile_name = os.path.join(procData_path, mat_file_name % name)
                            datafile = sio.loadmat(datafile_name)
                            for (j, cd) in enumerate(condition_name):
                                for i_rep in range(movement_rep):
                                    sheetname = cd + '_' + str(i_rep)
                                    data_x.append(datafile[sheetname])
                                    data_y.append(j)
                            del datafile
                            disk_visit_time += 1
                        else:
                            mem_itr_index = mem_random_index[
                                            mem_visit_time * samplePerSubject: (mem_visit_time + 1) * samplePerSubject]
                            for m_index in mem_itr_index:
                                data_x.append(mem_train_x[m_index])
                                data_y.append(mem_train_y[m_index])
                            mem_visit_time += 1
                        gc.collect()
                    shuffle_tmp = list(zip(data_x, data_y))
                    np.random.shuffle(shuffle_tmp)
                    data_x, data_y = zip(*shuffle_tmp)
                    del shuffle_tmp
                    gc.collect()
                    # data transform rwt method
                    data_x = np.array(data_x)
                    dnum, dx, dy, dz = data_x.shape
                    data_x = data_x.reshape(dnum, dx, dy, dz, 1)
                    data_y = to_categorical(np.array(data_y), num_classes=numClass)
                    yield (data_x, data_y)
                    del data_x, data_y
                    gc.collect()
                gc.collect()
        elif train_valid_test_flag in ['Valid']:
            while True:
                mem_random_index = [xx for xx in range(len(valid_y))]
                np.random.shuffle(mem_random_index)
                itr_time = int(len(valid_y) / batch_size)
                for itr in range(itr_time):
                    data_x, data_y = [], []
                    mem_itr_index = mem_random_index[itr * batch_size: (itr + 1) * batch_size]
                    for m_index in mem_itr_index:
                        data_x.append(valid_x[m_index])
                        data_y.append(valid_y[m_index])
                    # data transform rwt method
                    data_x = np.array(data_x)
                    dnum, dx, dy, dz = data_x.shape
                    data_x = data_x.reshape(dnum, dx, dy, dz, 1)
                    data_y = to_categorical(np.array(data_y), num_classes=numClass)
                    yield (data_x, data_y)
                    del data_x, data_y
                    gc.collect()
                gc.collect()
        elif train_valid_test_flag in ['Test']:
            test_num = len(subject_name)
            data_x = []
            data_y = []
            for ti in range(test_num):
                subname = subject_name[ti]
                datafile_name = os.path.join(procData_path, mat_file_name % subname)
                datafile = sio.loadmat(datafile_name)
                for (j, cd) in enumerate(condition_name):
                    for i_rep in range(movement_rep):
                        sheetname = cd + '_' + str(i_rep)
                        data_x.append(datafile[sheetname])
                        data_y.append(j)
                del datafile
                if (len(data_y) == batch_size) or (ti == (test_num - 1)):
                    # data transform rwt method
                    data_x = np.array(data_x)
                    dnum, dx, dy, dz = data_x.shape
                    data_x = data_x.reshape(dnum, dx, dy, dz, 1)
                    data_y = to_categorical(np.array(data_y), num_classes=numClass)
                    yield (data_x, data_y)
                    del data_x, data_y
                    data_x, data_y = [], []
                gc.collect()
            print('Test generator finished. ')
            del data_x, data_y
            gc.collect()
        else:
            subname = subject_name[0]
            datafile_name = os.path.join(procData_path, mat_file_name % subname)
            datafile = sio.loadmat(datafile_name)
            sheetname = condition_name[0] + '_0'
            one_data = datafile[sheetname]
            dx, dy, dz = one_data.shape
            one_data = one_data.reshape(dx, dy, dz, 1)
            print('Show data info: ', one_data.shape)
            del datafile, one_data
            gc.collect()
            return

    generator_CNN3D_MandD(train_name, 0, 'Info')

    batch_size = 20
    stepTrain = int(len(train_name) * samplePerSubject / batch_size)
    stepValid = int(len(valid_name) * samplePerSubject / batch_size)
    print('^^^^^^^^^^^^^^^^', stepTrain, stepValid)

    history = LossHistory()

    time_callback = TimeHistory()
    startTime = time.time()
    model.fit_generator(generator=generator_CNN3D_MandD(train_name, batch_size, 'Train'),
                        steps_per_epoch=stepTrain,
                        epochs=60, verbose=1, callbacks=[early_stop, checkpoint, lrate, time_callback, history],
                        validation_data=generator_CNN3D_MandD(valid_name, batch_size, 'Valid'),
                        validation_steps=stepValid)
    endTime = time.time()
    durTime = endTime - startTime
    print('&&& CNN3D Time: ', durTime)
    print("time_callback.times:", time_callback.times)
    print("time_callback.totaltime:", time_callback.totaltime)
    del mem_train_x, mem_train_y, valid_x, valid_y
    gc.collect()

    train_loss = history.losses['epoch']
    val_loss = history.val_loss['epoch']
    model_loss = []
    model_loss.append(train_loss)
    model_loss.append(val_loss)
    loss_file = '/home/kuang/project/hcp_clj/loss/cnn3d_995_%s_loss.txt' % str(round_index)
    np.savetxt(loss_file, model_loss, fmt="%.18f")

    model.load_weights(bestModelSavePath)

    # test
    all_pred, all_true_y = None, None
    predNoneFlag = True
    print('Start Predict.')
    for test_item in generator_CNN3D_MandD(test_name, batch_size, 'Test'):
        test_x, test_y = test_item[0], test_item[1]
        pred = model.predict(test_x)
        if predNoneFlag:
            all_pred = pred
            all_true_y = test_y
            predNoneFlag = False
        else:
            all_pred = np.concatenate((all_pred, pred), axis=0)
            all_true_y = np.concatenate((all_true_y, test_y), axis=0)
    print(all_pred.shape, all_true_y.shape)

    test_y_index, pred_index = [], []
    hit = 0
    for j in range(len(all_true_y)):
        ty_ind = int(np.where(all_true_y[j] == np.max(all_true_y[j]))[0])
        pd_ind = int(np.where(all_pred[j] == np.max(all_pred[j]))[0])
        print(ty_ind, '   ', pd_ind, '   ', all_true_y[j], all_pred[j])
        if ty_ind == pd_ind:
            hit += 1
        test_y_index.append(ty_ind)
        pred_index.append(pd_ind)
    acc_test = hit / len(all_true_y)
    print(acc_test)
    print(model.summary())

    test_acc, test_precision, test_recall, test_f1, test_cm = vote2calAcc(pred_index, test_y_index)

    print('cnn3d_method time: ', durTime)
    K.clear_session()
    # clear memory
    for xvar in locals().keys():
        if xvar not in ['test_acc', 'hit', 'test_precision', 'test_recall', 'test_f1', 'durTime', 'test_cm']:
            del locals()[xvar]
    gc.collect()
    return test_acc, hit, test_precision, test_recall, test_f1, durTime, test_cm

def sch_func(epoch, lr):
    init_lr = 0.0025
    drop = 0.5
    epochs_drop = 50
    new_lr = init_lr * (drop ** np.floor(epoch / epochs_drop))
    return new_lr

# split train and test set
def randomSplit_TrainAndTest(sl, k):
    np.random.seed(2018)
    np.random.shuffle(sl)
    print(sl)
    part_num = int(len(sl) / k)
    train_set = []
    test_set = []
    valid_set = []
    for i in range(k):
        test_part = []
        if i == k - 1:
            test_part = sl[i*part_num :]
        else:
            test_part = sl[i*part_num : (i+1)*part_num]
        test_valid_splitValue = int(len(test_part)/2)
        test_set.append(test_part[test_valid_splitValue:])
        valid_set.append(test_part[:test_valid_splitValue])
        train_set.append(list(set(sl) - set(test_part)))
    return train_set, test_set, valid_set

# evaluation method
def vote2calAcc(pred_cl, test_y_cl):
    pred_cl = np.array(pred_cl)
    test_y_cl = np.array(test_y_cl)
    n_test = len(test_y_cl)
    n_hit = 0
    for test_i in range(n_test):
        if test_y_cl[test_i] == pred_cl[test_i]:
            n_hit += 1
    test_acc = n_hit / n_test
    test_precision = sklearn.metrics.precision_score(test_y_cl, pred_cl, average='weighted')
    test_recall = sklearn.metrics.recall_score(test_y_cl, pred_cl, average='weighted')
    test_f1 = sklearn.metrics.f1_score(test_y_cl, pred_cl, average='weighted')
    test_cm = sklearn.metrics.confusion_matrix(test_y_cl, pred_cl).T
    print('***Log-result- ', test_acc, test_precision, test_recall, test_f1)
    print(test_cm)
    return test_acc, test_precision, test_recall, test_f1, test_cm

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()

    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))


if __name__ == '__main__':
    main()
