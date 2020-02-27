python="/Users/quan/anaconda3/bin/python3"
origin_train_data="../data/adult_train.txt"
origin_test_data="../data/adult_test.txt"
train_data="../data/train_file.txt"
test_data="../data/test_file.txt"
tree_model="../data/xgb.model"
lr_coef_mix_model="../data/lr_coef_mix_model"
xgb_mix_model="../data/xgb_mix_model"
feature_num_file="../data/feature_num"
model_type="lr_gbdt"    # "gbdt" or "lr_gbdt"

if [ -f $origin_train_data -a -f $origin_test_data ];then
  $python ana_train_data.py $origin_train_data $origin_test_data $train_data $test_data $feature_num_file
else
  echo "no origin data"
  exit
fi
if [ -f $train_data -a -f $feature_num_file ];then
  if [ $model_type == "gbdt" ];then
    $python train.py $train_data $feature_num_file $tree_model
  fi
  if [ $model_type == "lr_gbdt" ];then
    $python train.py $train_data $feature_num_file $xgb_mix_model $lr_coef_mix_model
  fi
else
  echo "no train file"
  exit
fi
if [ $model_type == "gbdt" ];then
  if [ -f $test_data -a -f $feature_num_file -a -f $tree_model ];then
    $python check.py $test_data $tree_model $feature_num_file
  else
    echo "no gbdt model file"
    exit
  fi
elif [ $model_type == "lr_gbdt" ]; then
  if [ -f $test_data -a -f $feature_num_file -a -f $xgb_mix_model -a -f $lr_coef_mix_model ];then
    $python check.py $test_data $xgb_mix_model $lr_coef_mix_model $feature_num_file
  else
    echo "no lr_gbdt model file"
    exit
  fi
else
  echo "wrong model type"
fi
