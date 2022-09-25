#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 4 ]; then
  echo "Usage: sh run_distribute_train_ascend.sh [TRAIN_DATA_DIR] [FILE_NAME] [TEACHER_MODEL] [RANK_TABLE_FILE]"
  exit 1
fi

DATAPATH=$1
echo "$DATAPATH"
FILENAME=$2
TPATH=$3
RANKTABLEPATH=$4

if [ ! -d $DATAPATH ]; then
  echo "error: TRAIN_DATA_DIR=$DATAPATH is not a directory"
  exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANKTABLEPATH

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
  export DEVICE_ID=$i
  export RANK_ID=$i
  rm -rf ./train_parallel$i
  mkdir ./train_parallel$i
  cp ../*.py ./train_parallel$i
  cp ../*.ckpt ./train_parallel$i
  cp *.sh ./train_parallel$i
  cp -r ../src ./train_parallel$i
  cp -r ../ckpt ./train_parallel$i
  cd ./train_parallel$i || exit
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  env >env.log
  nohup python csd_train.py \
        --device_target "Ascend" \
        --dir_data $DATAPATH \
        --data_test Set5 \
        --filename $FILENAME \
        --lr 0.0001 \
        --epochs 5000 \
        --ckpt_path $TPATH \
        --contra_lambda 200 > train.log 2>&1 &
  cd ..
done

