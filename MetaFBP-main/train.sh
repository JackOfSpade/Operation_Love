#!/bin/bash
if [ "$1" = 'Base-MAML' ]
then
  MODEL_OPTION='--model MAML'
elif [ "$1" = 'MetaFBP-R' ]
then
  MODEL_OPTION='--model DynamicMAML --dy-mode rebirth'
elif [ "$1" = 'MetaFBP-T' ]
then
  MODEL_OPTION='--model DynamicMAML --dy-mode tuning'
else
  echo 'the {model} arg should be one of "Base-MAML", "MetaFBP-R", or "MetaFBP-T"'
  echo "Usage: bash train.sh {arg1=model} {arg2=dataset}"
  exit 1
fi
BASE_OPTION="--backbone resnet18"
if [ "$2" = "PFBP-SCUT5500" ]
then
  DATASET_OPTION="--n-way 5 --dataset FBP5500"
elif [ "$2" = "PFBP-SCUT500" ]
then
  DATASET_OPTION="--n-way 3 --dataset FBPSCUT"

elif [ "$2" = "PFBP-US10K" ]
then
  DATASET_OPTION="--n-way 5 --dataset US10K"
else
  echo 'the {dataset} arg should be one of "PFBP-SCUT5500", "PFBP-SCUT500", or "PFBP-US10K"'
  echo "Usage: bash train.sh {arg1=model} {arg2=dataset}"
  exit 1
fi

MERGE_OPTION="$MODEL_OPTION $BASE_OPTION $DATASET_OPTION"

echo "$MERGE_OPTION"

# Use eval to correctly handle the merged options string
eval python3 train.py $MODEL_OPTION $BASE_OPTION $DATASET_OPTION --k-spt 1
eval python3 train.py $MODEL_OPTION $BASE_OPTION $DATASET_OPTION --k-spt 5
eval python3 train.py $MODEL_OPTION $BASE_OPTION $DATASET_OPTION --k-spt 10
eval python3 train.py $MODEL_OPTION $BASE_OPTION $DATASET_OPTION --k-spt 15

eval python3 test.py $MODEL_OPTION $BASE_OPTION $DATASET_OPTION --k-spts 1 5 10 15 --load-epoch last
