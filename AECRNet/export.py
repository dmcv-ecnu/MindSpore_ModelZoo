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
"""export checkpoint file into air, onnx, mindir models"""

import argparse
import numpy as np
import mindspore
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from src.args import args
from models.model import Dehaze

parser = argparse.ArgumentParser(description='MindSpore VNet Example')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="vnet", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
cfg = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
if cfg.device_target == "Ascend":
    context.set_context(device_id=cfg.device_id)

if __name__ == "__main__":

    # define VNet network
    network = Dehaze(3, 3, rgb_range=args.rgb_range)

    # load network checkpoint
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)

    # export network
    inputs = Tensor(np.ones([args.batch_size, 3, args.patch_size, args.patch_size]), mindspore.float32)
    export(network, inputs, file_name=cfg.file_name, file_format=cfg.file_format)
