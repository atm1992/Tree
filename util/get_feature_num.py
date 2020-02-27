# -*- coding: UTF-8 -*-

import os


def get_feature_num(feature_num_file):
    if not os.path.exists(feature_num_file):
        return 0
    with open(feature_num_file, "r") as f:
        for line in f:
            item = line.strip().split(":")
            if item[0] == "feature_num":
                return int(item[1])
    return 0


if __name__ == '__main__':
    print(get_feature_num("../data/feature_num"))
