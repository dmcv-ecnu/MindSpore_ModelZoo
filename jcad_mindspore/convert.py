import torch
import mindspore as ms
from mindspore.train import save_checkpoint, load_checkpoint
from mindspore import Tensor
from mindspore import Parameter
from lib.backbones.SwinTransformer import SwinTransformer
ms.context.set_context(device_target="GPU",device_id=1)
# def convert_torch_ms(pth_file, ms_model):
#     torch_para_dict = torch.load(pth_file, map_location=torch.device('cpu'))
#     print("*" * 10, "torch name list:")
#     torch_para_dict = torch_para_dict['model']
#     for k, v in torch_para_dict.items():
#         print(k)

    # ms_name_list = []
    # print("#" * 10, "ms name list:")
    # for name in ms_model.parameters_dict():
    #     print(name)
    #     ms_name_list.append(name)
    #
    # ms_params_list = []
    # for ms_name in ms_name_list:
    #     param_dict = {}
    #     param_dict['name'] = ms_name
    #     torch_name = convert_ms_name_to_torch(ms_name)  # TODO
    #     data = torch_para_dict[torch_name].numpy()
    #     param_dict['data'] = Tensor(data)
    #     ms_params_list.append(param_dict)
    # save_checkpoint(ms_params_list, "ms_weight.ckpt")
    # print("save ms weight sucess!")

# convert_torch_ms("/home/tkk/YZZ/data/backbone_ckpt/swin_base_patch4_window12_384_22kto1k.pth", "12345.ckpt")
pth_file = "/home/tkk/YZZ/data/backbone_ckpt/swin_base_patch4_window12_384_22kto1k.pth"
torch_para_dict = torch.load(pth_file, map_location=torch.device('cpu'))
torch_para_dict = torch_para_dict['model']

param_list = []
net = SwinTransformer(image_size=384, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12)
print("************************")
for x in net.get_parameters():
    param_list.append(x.name)
print(param_list)
print(len(param_list))
# torch_to_ms = {
#     'patch_embed.proj.weight': 'patch_embed.proj.weight',
#     'patch_embed.proj.bias': 'patch_embed.proj.bias',
#     'patch_embed.norm.weight': 'patch_embed.norm.beta',
#     'patch_embed.norm.bias': 'patch_embed.norm.gamma',
#     'layers.0.blocks.0.norm1.weight': '0.blocks.0.norm1.beta',
#     'layers.0.blocks.0.norm1.bias': '0.blocks.0.norm1.gamma',
#     'layers.0.blocks.0.attn.qkv.weight': '0.blocks.0.norm1.gamma',
#
#
#
# }
for key in list(torch_para_dict.keys()):  # 使用 list() 避免在循环中修改字典
    if 'qkv.weight' in key:
        # 拆分 qkv 权重
        total_weight = torch_para_dict[key]
        q_weight, k_weight, v_weight = total_weight.chunk(3, dim=0)

        # 更新字典
        torch_para_dict[key.replace('qkv.weight', 'q.weight')] = q_weight
        torch_para_dict[key.replace('qkv.weight', 'k.weight')] = k_weight
        torch_para_dict[key.replace('qkv.weight', 'v.weight')] = v_weight
        del torch_para_dict[key]  # 删除原始的 qkv.weight 键

    elif 'qkv.bias' in key:
        # 拆分 qkv 偏置
        total_bias = torch_para_dict[key]
        q_bias, k_bias, v_bias = total_bias.chunk(3, dim=0)

        # 更新字典
        torch_para_dict[key.replace('qkv.bias', 'q.bias')] = q_bias
        torch_para_dict[key.replace('qkv.bias', 'k.bias')] = k_bias
        torch_para_dict[key.replace('qkv.bias', 'v.bias')] = v_bias
        del torch_para_dict[key]  # 删除原始的 qkv.bias 键



ms_para_dict = {}
for key in torch_para_dict:
    # 去除 "layers." 前缀
    if key.startswith('layers.0'):
        new_key = key[7:]  # 从第8个字符开始截取
    elif key.startswith('layers.'):
        new_key = key[7:]
        new_key = '0.blocks.' + new_key
    else:
        new_key = key

    if key.endswith('_index'):
        new_key = new_key[:-6] + '.index'

    if new_key.endswith('relative_position.index'):
        new_key = new_key[:-23] + 'relative_bias.index'

    if new_key.endswith('relative_position_bias_table'):
        new_key = new_key[:-28] + 'relative_bias.relative_position_bias_table'


    # 对于包含“norm”的键，将其转换为对应的MindSpore格式
    if 'norm' in new_key:
        # 判断是权重还是偏置，并进行相应的转换
        if key.endswith('.weight'):
            new_key = new_key.replace('.weight', '.gamma')
        elif key.endswith('.bias'):
            new_key = new_key.replace('.bias', '.beta')
        else:
            # 如果不是权重或偏置，则保持原样
            new_key = new_key
    ms_para_dict[new_key] = ms.Tensor(torch_para_dict[key].numpy())
# print(ms_para_dict)

for key, tensor in ms_para_dict.items():
    # 将 Tensor 转换为 Parameter
    parameter = Parameter(tensor, name=key)
    ms_para_dict[key] = parameter

ms.save_checkpoint(ms_para_dict, "/home/tkk/YZZ/lib/backbones/swin384.ckpt")

# for key in ms_para_dict:
#     if key in param_list:
#         continue
#     else:
#         print(f"{key} is not in param_list")



