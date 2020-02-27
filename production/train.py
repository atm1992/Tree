# -*- coding: UTF-8 -*-

"""
train gbdt model
"""
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegressionCV as LRCV
from scipy.sparse import coo_matrix

import sys

sys.path.append("../")
from util.get_feature_num import get_feature_num


def get_train_data(train_file, feature_num_file):
    total_feature_num = get_feature_num(feature_num_file)
    train_label = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols=-1)
    feature_list = list(range(total_feature_num))
    train_feature = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols=feature_list)
    return train_feature, train_label


def train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate):
    """

    :param train_mat: train data and label
    :param tree_depth:
    :param tree_num: total tree num
    :param learning_rate: step_size
    :return: Booster
    """
    # "eta" —— 步长；"objective" —— 优化目标函数，也就是求解最优划分特征时一阶偏导和二阶偏导时所使用的函数
    param_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:squarederror", "verbosity": 0}
    bst = xgb.train(param_dict, train_mat, tree_num)
    # print(xgb.cv(param_dict,train_mat,tree_num,nfold=5,metrics="auc"))
    return bst


def choose_param():
    """

    :return: a list, as [(tree_depth,tree_num,learning_rate),...]
    """
    res_list = []
    # 参数的取值数组越长，训练时间也就越长。这里为了节省训练时间，因此数组里只写了一两个参数
    tree_depth_list = [4, 5, 6]
    tree_num_list = [10, 50, 100]
    learning_rate_list = [0.3, 0.5, 0.7]
    for ele_tree_depth in tree_depth_list:
        for ele_tree_num in tree_num_list:
            for ele_learning_rate in learning_rate_list:
                res_list.append((ele_tree_depth, ele_tree_num, ele_learning_rate))
    return res_list


def grid_search(train_mat):
    """
    选取GBDT模型的最优参数
    :param train_mat: train data and label
    """
    param_list = choose_param()
    for ele in param_list:
        (tree_depth, tree_num, learning_rate) = ele
        param_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:squarederror", "verbosity": 0}
        res = xgb.cv(param_dict, train_mat, tree_num, nfold=5, metrics="auc")
        auc_score = res.loc[tree_num - 1, ["test-auc-mean"]].values[0]
        print("tree_depth:{0}, tree_num:{1}, learning_rate:{2}, auc:{3}".format(tree_depth, tree_num, learning_rate,auc_score))


def train_tree_model(train_file, feature_num_file, tree_model_file):
    """

    :param train_file: data for train model
    :param feature_num_file: file to record feature total num
    :param tree_model_file: file to store model
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)
    # grid_search(train_mat)
    # sys.exit()
    # 通过网格搜索得到的最优参数
    tree_num = 10
    tree_depth = 6
    learning_rate = 0.3
    bst = train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate)
    bst.save_model(tree_model_file)


def get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth):
    """

    :param tree_leaf: 树模型的预测结果，是一个二维矩阵(数组)。行数为数据集中的样本个数(训练集为30162)，列数为树的棵数
    :param tree_num: total tree num
    :param tree_depth: total tree depth
    :return: a sparse matrix to record total train feature for lr part of mixed model
    """
    # total_node_num 表示一棵树中共有多少个节点，每棵树的深度都是一样的
    total_node_num = 2 ** (tree_depth + 1) - 1
    # leaf_node_num表示一棵树中有多少个叶节点
    leaf_node_num = 2 ** tree_depth
    # 一棵树中的非叶节点个数
    non_leaf_node_num = total_node_num - leaf_node_num
    # 总的特征维度(长度)。total_col_num为输出的稀疏矩阵的列数，total_row_num为稀疏矩阵的行数(数据集中的样本个数)
    total_col_num = leaf_node_num * tree_num
    total_row_num = len(tree_leaf)
    col = []
    row = []
    data = []
    base_row_idx = 0
    for one_result in tree_leaf:
        base_col_idx = 0
        # one_result as [15 18 15 15 23 27 13 17 28 21]
        for fix_idx in one_result:
            # 0<= leaf_idx < leaf_node_num
            # total_col_num = leaf_node_num*tree_num，存储每一棵树最终输出的叶节点索引位置leaf_idx
            leaf_idx = fix_idx - non_leaf_node_num
            leaf_idx = leaf_idx if leaf_idx >= 0 else 0
            # base_col_idx=0代表第一棵树，base_col_idx=leaf_node_num代表第二棵树
            col.append(base_col_idx + leaf_idx)
            row.append(base_row_idx)
            # 指定行、列位置的值为1
            data.append(1)
            base_col_idx += leaf_node_num
        base_row_idx += 1
        total_feature_list = coo_matrix((data, (row, col)), shape=(total_row_num, total_col_num))
    return total_feature_list


def get_mix_model_tree_info():
    tree_num = 10
    tree_depth = 4
    learning_rate = 0.3
    result = (tree_num, tree_depth, learning_rate)
    return result

def train_gbdt_and_lr_model(train_file, feature_num_file, mix_tree_model_file, mix_lr_model_file):
    """

    :param train_file: file for training model
    :param feature_num_file: file to store total feature len
    :param mix_tree_model_file: tree part of the mix model
    :param mix_lr_model_file: lr part of the mix model
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)
    tree_num, tree_depth, learning_rate = get_mix_model_tree_info()
    bst = train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate)
    bst.save_model(mix_tree_model_file)
    # tree_leaf是一个二维矩阵，行数为数据集中的样本个数(训练集为30162)，列数为树的棵数
    # 表示每个样本在每棵树中所预测的叶节点index（该叶节点在整棵树节点中的下标index，根节点下标为0）
    tree_leaf = bst.predict(train_mat, pred_leaf=True)
    # print(tree_leaf[0])
    # 输出树中最大的叶节点索引，节点索引从0开始，即 树中共有np.max(tree_leaf) +1 个节点
    # print(np.max(tree_leaf))
    total_feature_list = get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth)
    lr_cf = LRCV(Cs=[1], penalty="l2", dual=False, tol=1e-4, max_iter=500, cv=5).fit(total_feature_list, train_label)
    # 得到一个5行3列的二维数组
    scores = list(lr_cf.scores_.values())[0]
    # 查看每一个正则化参数(1、0.1、0.01)所对应的5折交叉验证的平均准确率
    # 由结果可知，正则化参数为0.01时，模型的准确率最高
    print("diff:", ",".join([str(ele) for ele in scores.mean(axis=0)]))
    # 各个正则化参数的总平均准确率
    print("accuracy:{0} (+-{1:.3f})".format(scores.mean(), scores.std() * 2))

    # 除了上述查看模型的准确率，还需要关心模型的AUC，scoring="roc_auc"
    lr_cf = LRCV(Cs=[1], penalty="l2", dual=False, tol=1e-4, max_iter=500, cv=5, scoring="roc_auc").fit(total_feature_list, train_label)
    scores = list(lr_cf.scores_.values())[0]
    # 由结果可知，正则化参数为1时，模型的AUC最高。与上述的准确率最优参数不一致，这里倾向于选择AUC最高的参数1
    print("diff:", ",".join([str(ele) for ele in scores.mean(axis=0)]))
    # 各个正则化参数的总平均AUC
    print("AUC:{0} (+-{1:.3f})".format(scores.mean(), scores.std() * 2))

    with open(mix_lr_model_file, "w+") as f:
        f.write(",".join([str(ele) for ele in lr_cf.coef_[0]]))


if __name__ == '__main__':
    # train_tree_model("../data/train_file.txt","../data/feature_num","../data/xgb.model")
    # train_gbdt_and_lr_model("../data/train_file.txt", "../data/feature_num", "../data/xgb_mix_model","../data/lr_coef_mix_model")
    if len(sys.argv) == 4:
        train_file = sys.argv[1]
        feature_num_file = sys.argv[2]
        tree_model = sys.argv[3]
        train_tree_model(train_file,feature_num_file,tree_model)
    elif len(sys.argv) == 5:
        train_file = sys.argv[1]
        feature_num_file = sys.argv[2]
        xgb_mix_model = sys.argv[3]
        lr_coef_mix_model = sys.argv[4]
        train_gbdt_and_lr_model(train_file,feature_num_file,xgb_mix_model,lr_coef_mix_model)
    else:
        print("train gbdt model usage: python xx.py train_data feature_num_file tree_model")
        print("train lr_gbdt model usage: python xx.py train_data feature_num_file xgb_mix_model lr_coef_mix_model")
        sys.exit()
