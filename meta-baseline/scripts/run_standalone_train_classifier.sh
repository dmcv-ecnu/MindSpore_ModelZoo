#!/usr/bin/env bash
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

echo "=========================================================================================="
echo "Please run the script as: "
echo "bash run_train_classifier.sh [device_target] [ROOT_PATH]"

echo "========================================================================================="
export DEVICE=$1
ROOT_PATH=$2
python ./train_classifier.py \
    --run_offline "True" \
    --device_target $DEVICE \
    --root_path $ROOT_PATH > log.txt 2>&1 &
