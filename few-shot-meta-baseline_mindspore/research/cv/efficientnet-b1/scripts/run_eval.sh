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

if [ $# != 2 ]
then
    echo "Using: bash scripts/run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

DATA_PATH=$(get_real_path $1)                     # dataset_path
CHECKPOINT_PATH=$(get_real_path $2)               # checkpoint_path

if [ ! -d $DATA_PATH ]
then
    echo "error: DATA_PATH=$DATA_PATH is not a directory."
    exit 1
fi

if [ ! -f $CHECKPOINT_PATH ]
then
    echo "error: TRAIN_PATH=$TRAIN_PATH is not a directory."
    exit 1
fi

python ./eval.py  \
    --checkpoint_path=$CHECKPOINT_PATH \
    --data_path=$DATA_PATH \
    --model efficientnet-b1 \
    --modelarts False \
    --device_target Ascend > eval_log 2>&1 &
