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

if [ $# != 3 ]
then
    echo "Usage: $0 [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)

if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -f $PATH2 ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=$3
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "train_sysu_indoor" ];
then
    rm -rf ./train_sysu_indoor
fi
mkdir ./train_sysu_indoor
cp ../*.py ./train_sysu_indoor
cp -r ../src ./train_sysu_indoor
cd ./train_sysu_indoor || exit
env > env.log
echo "start training for device $DEVICE_ID"

python train.py \
--MSmode GRAPH_MODE \
--dataset SYSU \
--data_path $PATH1 \
--optim adam \
--lr 0.0035 \
--device_target Ascend \
--device_id $DEVICE_ID \
--pretrain $PATH2 \
--loss_func id+tri \
--sysu_mode indoor \
--epoch 60 \
--print_per_step 100 &> log &
cd ..