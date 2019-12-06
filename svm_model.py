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
from keras.utils.np_utils import to_categorical
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA, IncrementalPCA
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

        result = pcaSVM_method(train_name, test_name)

        gc.collect()

        all_result_value.append(result[:-1])

    print('each : test_acc, num_hit, test_precision, test_recall, test_f1')
    print('pcaSVM_method : ', all_result_value)

    print('ave : test_acc, num_hit, test_precision, test_recall, test_f1')
    print('pcaSVM_method : ', np.mean(all_result_value, axis=0))
    print('pcaSVM_method std: ', np.std(all_result_value, axis=0))

    print('Finished.')

def pcaSVM_method(train_name, test_name):
    print('*** Run PCA+SVM method ...')

    def generator0_PCASVM(subject_name0, batch_size):
        subject_name = np.array(subject_name0)
        subject_num = len(subject_name)
        data_x = []
        data_y = []
        for ti in range(subject_num):
            subname = subject_name[ti]
            datafile_name = os.path.join(procData_path, mat_file_name % subname)
            datafile = sio.loadmat(datafile_name)
            for (j, cd) in enumerate(condition_name):
                for i_rep in range(movement_rep):
                    sheetname = cd + '_' + str(i_rep)
                    data_x.append(datafile[sheetname].flatten())
                    data_y.append(j)
            del datafile
            if (len(data_y) >= batch_size) or (ti == (subject_num - 1)):
                data_x = np.array(data_x)
                data_y = np.array(data_y)
                yield (data_x, data_y)
                del data_x, data_y
                data_x, data_y = [], []
            gc.collect()
        print('... generator finished ... ')
        del data_x, data_y
        gc.collect()

    original_dim = DIMX*DIMY*DIMZ
    ratio = 0.0002
    reduceDim = 500    # int(original_dim * ratio)
    generator_batch_size = 500             #200
    ipca_batch_size = 20

    ipca = IncrementalPCA(n_components=reduceDim, whiten=False, copy=False, batch_size=ipca_batch_size)
    # fit
    for (batch_train_x, batch_train_y) in generator0_PCASVM(train_name, generator_batch_size):
        print(batch_train_x.shape, batch_train_y.shape)
        ipca.partial_fit(batch_train_x)
        gc.collect()
    ipca_explainVar = ipca.explained_variance_ratio_
    ipca_comp_vetor = ipca.components_
    esum = 0
    for zz in range(len(ipca_explainVar)):
        esum += ipca_explainVar[zz]

    # transform train
    train_x, train_y = None, None
    trainNoneFlag = True
    for (batch_train_x, batch_train_y) in generator0_PCASVM(train_name, generator_batch_size):
        rd_train_x = ipca.transform(batch_train_x)
        if trainNoneFlag:
            train_x = rd_train_x
            train_y = batch_train_y
            trainNoneFlag = False
            print(batch_train_x.shape, 'after ipca 1st train batch shape: ', rd_train_x.shape)
        else:
            train_x = np.concatenate((train_x, rd_train_x), axis=0)
            train_y = np.concatenate((train_y, batch_train_y), axis=0)
        gc.collect()
    print(train_x.shape, train_y.shape)

    # train SVM
    clf = svm.LinearSVC()
    clf.fit(train_x, train_y)
    train_score = clf.score(train_x, train_y)
    del train_x, train_y
    gc.collect()

    # transform test
    test_x, test_y = None, None
    testNoneFlag = True
    for (batch_test_x, batch_test_y) in generator0_PCASVM(test_name, generator_batch_size):
        rd_test_x = ipca.transform(batch_test_x)
        if testNoneFlag:
            test_x = rd_test_x
            test_y = batch_test_y
            testNoneFlag = False
        else:
            test_x = np.concatenate((test_x, rd_test_x), axis=0)
            test_y = np.concatenate((test_y, batch_test_y), axis=0)
        gc.collect()
    print(test_x.shape, test_y.shape)

    test_score = clf.score(test_x, test_y)
    pred = clf.decision_function(test_x)

    print(test_y.shape, pred.shape)
    test_y_index, pred_index = [], []
    hit = 0
    for j in range(len(test_y)):
        ty_ind = test_y[j]
        pd_ind = int(np.where(pred[j] == np.max(pred[j]))[0])
        print(ty_ind, '   ', pd_ind, '   ', test_y[j], pred[j])
        if ty_ind == pd_ind:
            hit += 1
        test_y_index.append(ty_ind)
        pred_index.append(pd_ind)
    acc_test = hit / len(test_y)
    print(train_score, test_score, acc_test)

    test_acc, test_precision, test_recall, test_f1, test_cm = vote2calAcc(pred_index, test_y_index)
    # clear memory
    for xvar in locals().keys():
        if xvar not in ['test_acc', 'hit', 'test_precision', 'test_recall', 'test_f1', 'test_cm']:
            del locals()[xvar]
    gc.collect()

    return test_acc, hit, test_precision, test_recall, test_f1, test_cm

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

if __name__ == '__main__':
    main()
