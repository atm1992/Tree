# -*- coding: UTF-8 -*-

"""
test the performance of the model in the test data for gbdt and gbdt mix lr
"""
import math
import operator

import numpy as np
import xgboost as xgb
from scipy.sparse import csc_matrix
import sys

sys.path.append("../")
from util.get_feature_num import get_feature_num
from production import train as TA


def get_test_data(test_file, feature_num_file):
    """

    :param test_file: file to check performance
    :return: two np array: test_feature, test_label
    """
    total_feature_num = get_feature_num(feature_num_file)
    test_label = np.genfromtxt(test_file, dtype=np.int32, delimiter=",", usecols=-1)
    test_feature_list = list(range(total_feature_num))
    test_feature = np.genfromtxt(test_file, dtype=np.int32, delimiter=",", usecols=test_feature_list)
    return test_feature, test_label


def predict_by_tree(test_feature, tree_model):
    """
    predict by gbdt model
    :param test_feature:
    :param tree_model:
    :return:
    """
    predict_list = tree_model.predict(xgb.DMatrix(test_feature))
    return predict_list


def get_auc(predict_list, test_label):
    """
    auc = (sum(pos_idx) - n_pos*(n_pos+1)/2)/(n_pos*n_neg)
    sum(pos_idx) —— 先按照预测概率将所有样本(包括负样本)进行升序，将所有的正样本所处的位置index累加求和，预测概率最低的那个样本的index为1
    n_pos 为正样本的个数；n_neg 为负样本的个数
    :param predict_list:
    :param test_label:
    :return:
    """
    total_list = []
    for i in range(len(predict_list)):
        predict_score = predict_list[i]
        label = test_label[i]
        total_list.append((label, predict_score))
    n_pos, n_neg = 0, 0
    count = 1
    total_pos_idx = 0
    for label, _ in sorted(total_list, key=operator.itemgetter(1)):
        if label == 0:
            n_neg += 1
        else:
            n_pos += 1
            total_pos_idx += count
        count += 1
    auc_score = (total_pos_idx - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    print("auc:{:.5f}".format(auc_score))


def get_accuracy(predict_list, test_label):
    score_thr = 0.6
    acc_num = 0
    for i in range(len(predict_list)):
        predict_score = predict_list[i]
        if predict_score >= score_thr:
            predict_label = 1
        else:
            predict_label = 0
        if predict_label == test_label[i]:
            acc_num += 1
    acc_score = acc_num / len(predict_list)
    print("accuracy:{:.5f}".format(acc_score))


def run_check_core(test_feature, test_label, model, score_func):
    """

    :param test_feature:
    :param test_label:
    :param model: tree model
    :param score_func: use different model to predict
    """
    predict_list = score_func(test_feature, model)
    # print(predict_list[:10])
    get_auc(predict_list, test_label)
    get_accuracy(predict_list, test_label)


def run_check(test_file, tree_model_file, feature_num_file):
    """

    :param test_file: file to test performance
    :param tree_model_file: gbdt model file
    :param feature_num_file: file to store feature num
    """
    test_feature, test_label = get_test_data(test_file, feature_num_file)
    tree_model = xgb.Booster(model_file=tree_model_file)
    # predict_by_tree 是一个打分函数
    run_check_core(test_feature, test_label, tree_model, predict_by_tree)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict_by_lr_gbdt(test_feature, mix_tree_model, mix_lr_coef, tree_info):
    """
    predict by mix model
    :param test_feature:
    :param mix_tree_model:
    :param mix_lr_coef:
    :param tree_info:
    :return:
    """
    tree_leaf = mix_tree_model.predict(xgb.DMatrix(test_feature), pred_leaf=True)
    tree_num, tree_depth, learning_rate = tree_info
    total_feature_list = TA.get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth)
    result_list = np.dot(csc_matrix(mix_lr_coef), total_feature_list.tocsc().T).toarray()[0]
    sigmoid_ufunc = np.frompyfunc(sigmoid, 1, 1)
    return sigmoid_ufunc(result_list)


def run_check_lr_gbdt_core(test_feature, test_label, mix_tree_model, mix_lr_coef, tree_info, score_func):
    """
    core function to test the performance of the mix model
    :param test_feature:
    :param test_label:
    :param mix_tree_model:
    :param mix_lr_coef:
    :param tree_info: 一个包含 tree_num, tree_depth, learning_rate 的三元组
    :param score_func: different score function
    """
    predict_list = score_func(test_feature, mix_tree_model, mix_lr_coef, tree_info)
    get_auc(predict_list, test_label)
    get_accuracy(predict_list, test_label)


def run_check_lr_gbdt(test_file, tree_mix_model_file, lr_coef_mix_model_file, feature_num_file):
    """

    :param test_file:
    :param tree_mix_model_file: tree part of the mix model
    :param lr_coef_mix_model_file: lr part of the mix model
    :param feature_num_file:
    """
    test_feature, test_label = get_test_data(test_file, feature_num_file)
    # 加载混合模型中的GBDT模型
    mix_tree_model = xgb.Booster(model_file=tree_mix_model_file)
    mix_lr_coef = np.genfromtxt(lr_coef_mix_model_file, dtype=np.float32, delimiter=",")
    tree_info = TA.get_mix_model_tree_info()
    # predict_by_lr_gbdt是一个预测打分函数
    run_check_lr_gbdt_core(test_feature, test_label, mix_tree_model, mix_lr_coef, tree_info, predict_by_lr_gbdt)


if __name__ == '__main__':
    # run_check("../data/test_file.txt", "../data/xgb.model", "../data/feature_num")
    # run_check_lr_gbdt("../data/test_file.txt", "../data/xgb_mix_model", "../data/lr_coef_mix_model","../data/feature_num")
    if len(sys.argv) == 4:
        test_data = sys.argv[1]
        tree_model = sys.argv[2]
        feature_num_file = sys.argv[3]
        run_check(test_data,tree_model,feature_num_file)
    elif len(sys.argv) == 5:
        test_data = sys.argv[1]
        xgb_mix_model = sys.argv[2]
        lr_coef_mix_model = sys.argv[3]
        feature_num_file = sys.argv[4]
        run_check_lr_gbdt(test_data,xgb_mix_model,lr_coef_mix_model,feature_num_file)
    else:
        print("check gbdt model usage: python xx.py test_data tree_model feature_num_file")
        print("check lr_gbdt model usage: python xx.py test_data xgb_mix_model lr_coef_mix_model feature_num_file")
        sys.exit()
