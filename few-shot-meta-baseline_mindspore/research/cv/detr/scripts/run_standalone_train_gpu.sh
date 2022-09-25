#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

if [ $# != 5 ]
then
    echo "Usage: bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [BACKBONE_PRETRAIN] [DATASET_PATH]"
exit 1
fi

DEVICE_ID=$1
CFG_PATH=$2
SAVE_PATH=$3
BACKBONE_PRETRAIN=$4
DATASET_PATH=$5

export DEVICE_ID="$DEVICE_ID"

if [ -d "$SAVE_PATH" ];
then
    rm -rf "$SAVE_PATH"
fi
mkdir -p "$SAVE_PATH"

cp "$CFG_PATH" "$SAVE_PATH"

python train.py \
  --is_distributed=0 \
  --device_target=GPU \
  --config_path="$CFG_PATH" \
  --backbone_pretrain="$BACKBONE_PRETRAIN" \
  --coco_path="$DATASET_PATH" \
  --save_path="$SAVE_PATH" > "$SAVE_PATH"/log.txt 2>&1 &
