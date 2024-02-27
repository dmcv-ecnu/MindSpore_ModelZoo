from data_loader import TrainWholeImageDataset, MVTecImageAnomTrainDataset
import os
import numpy as np
from dsr_model import SubspaceRestrictionModule, ImageReconstructionNetwork, AnomalyDetectionModule, UpsamplingModule
from discrete_model import DiscreteLatentModel
import sys
from loss import FocalLoss
from sklearn.metrics import roc_auc_score, average_precision_score
from data_loader_test import TestMVTecDataset
import time

from mindspore import context
import mindspore
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset
from mindspore import context

def detach_temp(original_tensor):
    detached_tensor = mindspore.Tensor(original_tensor)
    detached_tensor.requires_grad = False  
    return detached_tensor

def clone_temp(original_tensor):
    cloned_tensor = mindspore.Tensor(original_tensor, dtype=original_tensor.dtype)
    cloned_tensor.requires_grad = True

# def forward_fn(data, label):
#     logits = model(data)
#     loss = loss_fn(logits, label)
#     return loss, logits     
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def generate_fake_anomalies_joined(features, embeddings, memory_torch_original, mask, diversity=1.0, strength=None):
    random_embeddings = ops.zeros((embeddings.shape[0], embeddings.shape[2] * embeddings.shape[3], memory_torch_original.shape[1]))
    # inputs = features.permute(0, 2, 3, 1).contiguous()
    inputs = features.permute(0, 2, 3, 1) # question

    for k in range(embeddings.shape[0]):
        memory_torch = memory_torch_original
        flat_input = inputs[k].view(-1, memory_torch.shape[1])

        distances_b = (ops.sum(flat_input ** 2, dim=1, keepdim=True)
                     + ops.sum(memory_torch ** 2, dim=1)
                     - 2 * ops.matmul(flat_input, memory_torch.t()))
        percentage_vectors = strength[k]
        topk = max(1, min(int(percentage_vectors * memory_torch.shape[0]) + 1, memory_torch.shape[0] - 1))
        values, topk_indices = ops.topk(distances_b, topk, dim=1, largest=False)
        topk_indices = topk_indices[:, int(memory_torch.shape[0] * 0.05):]
        topk = topk_indices.shape[1]

        random_indices_hik = ops.randint(0, topk, size=(topk_indices.shape[0],))
        random_indices_t = topk_indices[ops.arange(random_indices_hik.shape[0]),random_indices_hik]
        random_embeddings[k] = memory_torch[random_indices_t,:]
    
    random_embeddings = random_embeddings.reshape((random_embeddings.shape[0],embeddings.shape[2],embeddings.shape[3],random_embeddings.shape[2]))
    random_embeddings_tensor = random_embeddings.permute(0,3,1,2) # question

    down_ratio_y = int(mask.shape[2]/embeddings.shape[2])
    down_ratio_x = int(mask.shape[3]/embeddings.shape[3])
    anomaly_mask = mindspore.ops.adaptive_max_pool2d(mask, (down_ratio_y, down_ratio_x)).float()   # question

    anomaly_embedding = anomaly_mask * random_embeddings_tensor + (1.0 - anomaly_mask) * embeddings

    return anomaly_embedding


def train_upsampling_module(model, sub_res_model_hi, sub_res_model_lo, model_decode, decoder_seg, obj_name, mvtec_path, out_path, lr, batch_size, epochs, anom_par):
    run_name = 'dsr_' + str(lr) + '_' + str(epochs) + '_bs' + str(batch_size) + "_anom_par" + str(anom_par) + "_"

    embedding_dim = 128
    # model.eval()
    model.set_train(False)
    sub_res_model_hi.set_train(False)
    sub_res_model_lo.set_train(False)
    decoder_seg.set_train(False)
    model_decode.set_train(False)

    model_upsample = UpsamplingModule(embedding_size=embedding_dim)
    # model_upsample.cuda()
    model_upsample.set_train(True)

    optimizer = mindspore.nn.Adam(model_upsample.trainable_params(), learning_rate=lr)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[int(epochs*0.80), int(epochs*0.90)],gamma=0.2, last_epoch=-1)

    loss_focal = FocalLoss()

    dataset = MVTecImageAnomTrainDataset(mvtec_path + obj_name + "/train/good/", resize_shape=[256, 256])
    dataloader = GeneratorDataset(dataset, shuffle=True, num_parallel_workers=12)
    dataloader = dataloader.batch(batch_size)

    n_iter = 0.0

    segment_loss_avg = 0.0

    for epoch in range(epochs//2):
        start_time = time.time()
        for i_batch, sample_batched in enumerate(dataloader):

            input_image_aug = sample_batched["augmented_image"].cuda()
            anomaly_mask = sample_batched["anomaly_mask"].cuda()

            optimizer.zero_grad()

            loss_b, loss_t, data_recon, embeddings_t, embeddings = model(input_image_aug)

            data_recon = data_recon.detach()
            embeddings = embeddings.detach()
            embeddings_t = embeddings_t.detach()

            embedder = model._vq_vae_bot
            embedder_top = model._vq_vae_top

            anomaly_embedding_copy = embeddings.clone()
            anomaly_embedding_top_copy = embeddings_t.clone()
            recon_feat, recon_embeddings, _ = sub_res_model_hi(anomaly_embedding_copy, embedder)
            recon_feat_top, recon_embeddings_top, loss_b_top = sub_res_model_lo(anomaly_embedding_top_copy,
                                                                                embedder_top)

            up_quantized_recon_t = model.upsample_t(recon_embeddings_top)
            quant_join = ops.cat((up_quantized_recon_t, recon_embeddings), axis=1)
            recon_image_recon = model_decode(quant_join)

            ################################################
            up_quantized_embedding_t = model.upsample_t(embeddings_t)
            quant_join_real = ops.cat((up_quantized_embedding_t, embeddings), axis=1)
            recon_image = model._decoder_b(quant_join_real)
            
            temp_softmax = mindspore.nn.Softmax()
            out_mask = decoder_seg(recon_image_recon, recon_image)
            out_mask_sm = temp_softmax(out_mask, dim=1)
            refined_mask = model_upsample(recon_image_recon, recon_image, out_mask_sm)
            refined_mask_sm = temp_softmax(refined_mask, dim=1)

            segment_loss = loss_focal(refined_mask_sm, anomaly_mask)

            loss = segment_loss
            loss.backward()
            optimizer.step()

            segment_loss_avg = segment_loss_avg * 0.95 + 0.05 * segment_loss.item()

            n_iter +=1

        scheduler.step()

        if epoch % 5 == 0:
            torch.save(model_upsample.state_dict(), out_path+"checkpoints/"+run_name+"_upsample.pckl")




def train_on_device(obj_names, mvtec_path, out_path, lr, batch_size, epochs):
    run_name_pre = 'vq_model_pretrained_128_4096'
    num_hiddens = 128
    num_residual_hiddens = 64
    num_residual_layers = 2
    embedding_dim = 128
    num_embeddings = 4096
    commitment_cost = 0.25
    decay = 0.99
    anom_par = 0.2
    
    context.set_context(device_target="GPU", device_id=0)
    # context.set_context(enable_graph_kernel=True)
    # context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")

    # Load the pretrained discrete latent model used.
    model = DiscreteLatentModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay)
    # model.load_state_dict(torch.load("./checkpoints/" + run_name_pre + ".pckl", map_location='cuda:0')) # quesiton:这里明显是别人train好的模型
    # model.eval() question:不知道

    # Modules using the codebooks K_hi and K_lo for feature quantization
    embedder_hi = model._vq_vae_bot
    embedder_lo = model._vq_vae_top

    for obj_name in obj_names:
        run_name = 'dsr_'+str(lr)+'_'+str(epochs)+'_bs'+str(batch_size)+"_anom_par"+str(anom_par)+"_"

        # Define the subspace restriction modules - Encoder decoder networks
        sub_res_model_lo = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_model_hi = SubspaceRestrictionModule(embedding_size=embedding_dim)
        # sub_res_model_lo.cuda()
        # sub_res_model_hi.cuda()

        # Define the anomaly detection module - UNet-based network
        decoder_seg = AnomalyDetectionModule(embedding_size=embedding_dim)
        # decoder_seg.cuda()
        # decoder_seg.apply(weights_init)   # question:先不管这个


        # Image reconstruction network reconstructs the image from discrete features.
        # It is trained for a specific object
        model_decode = ImageReconstructionNetwork(embedding_dim * 2, num_hiddens, num_residual_layers, num_residual_hiddens)
        # model_decode.cuda()
        # model_decode.apply(weights_init)
        
        group_params = [{'params': sub_res_model_lo.trainable_params(), 'lr': lr},
                        {'params': sub_res_model_hi.trainable_params(), 'lr': lr},
                        {'params': model_decode.trainable_params(), 'lr': lr},
                        {'params': decoder_seg.trainable_params(), 'lr': lr}]
        group_params = [{'params': sub_res_model_lo.trainable_params(), 'lr': lr}]
        optimizer = mindspore.nn.Adam(group_params, learning_rate=0.1, weight_decay=0.0, use_lazy=False, use_offload=False)
        # optimizer = mindspore.nn.Adam([
        #                               {"params": sub_res_model_lo.trainable_params(), "lr": lr},
        #                               {"params": sub_res_model_hi.trainable_params(), "lr": lr},
        #                               {"params": model_decode.trainable_params(), "lr": lr},
        #                               {"params": decoder_seg.trainable_params(), "lr": lr}])   # question: 在小实验中这里出现了很怪的错误，raise ValueError("The value {} , its name '{}' already exist

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(epochs*0.80)], gamma=0.1, last_epoch=-1)
        # scheduler = mindspore.nn.piecewise_constant_lr()    # question: 这里要重新自己写
        
        loss_focal = FocalLoss()

        dataset = TrainWholeImageDataset(mvtec_path + obj_name + "/train/good/", resize_shape=[256, 256], perlin_augment=True)
        dataloader = mindspore.dataset.GeneratorDataset(dataset, column_names=["image", "mask", "is_normal", "idx"], num_parallel_workers=1, shuffle=True)   # question: column_names
        dataloader = dataloader.batch(batch_size=batch_size)

        n_train = len(dataset)

        n_iter = 0.0
        start_time = time.time()
    
        for epoch in range(1):
            print("Epoch ", epoch)
            
            for i_batch, sample_batched in enumerate(dataloader):

                # in_image = sample_batched["image"]  # question: 省去了cuda()
                # anomaly_mask = sample_batched["mask"] 
                
                in_image, anomaly_mask, is_noraml, idx = sample_batched
                
                # optimizer.zero_grad() # question: 

                # with torch.no_grad():     # question: 这里其实不太懂
                anomaly_strength_lo = (ops.rand(in_image.shape[0]) * (1.0-anom_par) + anom_par)
                anomaly_strength_hi = (ops.rand(in_image.shape[0]) * (1.0-anom_par) + anom_par)

                
                # Extract features from the discrete model
                enc_b = model._encoder_b(in_image)
                enc_t = model._encoder_t(enc_b)
                zt = model._pre_vq_conv_top(enc_t)

                
                # Quantize the extracted features
                loss_t, quantized_t, perplexity_t, encodings_t = embedder_lo(zt)
                # loss_t, perplexity_t, encodings_t = embedder_lo(zt)
                quantized = quantized.permute(0, 3, 1, 2)
                print(loss_t)
                print("out succcessfully")
                # Generate feature-based anomalies on F_lo
                # print("hello !") # question: 这里embedder_to有时会出错，不太懂？
                # anomaly_embedding_lo = generate_fake_anomalies_joined(zt, quantized_t,
                #                                                         embedder_lo._embedding_weights,
                #                                                         anomaly_mask, strength=anomaly_strength_lo)
                # Upsample the extracted quantized features and the quantized features augmented with anomalies

                
                # up_quantized_t = model.upsample_t(anomaly_embedding_lo)
                # print("2")
                # up_quantized_t_real = model.upsample_t(quantized_t)
                # feat = ops.cat((enc_b, up_quantized_t), axis=1)
                # feat_real = ops.cat((enc_b, up_quantized_t_real), axis=1)
                # zb = model._pre_vq_conv_bot(feat)
                # zb_real = model._pre_vq_conv_bot(feat_real)
                # # Quantize the upsampled features - F_hi
                # loss_b, quantized_b, perplexity_b, encodings_b = embedder_hi(zb)
                # loss_b, quantized_b_real, perplexity_b, encodings_b = embedder_hi(zb_real)

                # # Generate feature-based anomalies on F_hi
                # anomaly_embedding = generate_fake_anomalies_joined(zb, quantized_b,
                #                                                         embedder_hi._embedding_weights , anomaly_mask
                #                                                         , strength=anomaly_strength_hi)
                # print("hello 2")   
                    
                # use_both = ops.randint(0, 2,(in_image.shape[0],1,1,1)).float()
                # use_lo = ops.randint(0, 2,(in_image.shape[0],1,1,1)).float()
                # use_hi = (1 - use_lo)
                # anomaly_embedding_hi_usebot = generate_fake_anomalies_joined(zb_real,
                #                                                         quantized_b_real,
                #                                                         embedder_hi._embedding_weights ,
                #                                                         anomaly_mask, strength=anomaly_strength_hi)
                # anomaly_embedding_lo_usebot = quantized_t
                # anomaly_embedding_hi_usetop = quantized_b_real
                # anomaly_embedding_lo_usetop = anomaly_embedding_lo
                # anomaly_embedding_hi_not_both =  use_hi * anomaly_embedding_hi_usebot + use_lo * anomaly_embedding_hi_usetop
                # anomaly_embedding_lo_not_both =  use_hi * anomaly_embedding_lo_usebot + use_lo * anomaly_embedding_lo_usetop
                # # anomaly_embedding_hi = (anomaly_embedding * use_both + anomaly_embedding_hi_not_both * (1.0 - use_both)).detach().clone()
                # # anomaly_embedding_lo = (anomaly_embedding_lo * use_both + anomaly_embedding_lo_not_both * (1.0 - use_both)).detach().clone()
                # anomaly_embedding_hi = clone_temp(detach_temp(anomaly_embedding * use_both + anomaly_embedding_hi_not_both * (1.0 - use_both)))
                # anomaly_embedding_lo = clone_temp(detach_temp(anomaly_embedding_lo * use_both + anomaly_embedding_lo_not_both * (1.0 - use_both)))
                
                # anomaly_embedding_hi_copy = clone_temp(anomaly_embedding_hi)
                # anomaly_embedding_lo_copy = clone_temp(anomaly_embedding_lo)   # question

                # # Restore the features to normality with the Subspace restriction modules
                # recon_feat_hi, recon_embeddings_hi, loss_b = sub_res_model_hi(anomaly_embedding_hi_copy, embedder_hi)
                # recon_feat_lo, recon_embeddings_lo, loss_b_lo = sub_res_model_lo(anomaly_embedding_lo_copy, embedder_lo)

                # # Reconstruct the image from the anomalous features with the general appearance decoder
                # up_quantized_anomaly_t = model.upsample_t(anomaly_embedding_lo)
                # quant_join_anomaly = ops.cat((up_quantized_anomaly_t, anomaly_embedding_hi), axis=1)
                # recon_image_general = model._decoder_b(quant_join_anomaly)


                # # Reconstruct the image from the reconstructed features
                # # with the object-specific image reconstruction module
                # up_quantized_recon_t = model.upsample_t(recon_embeddings_lo)
                # quant_join = ops.cat((up_quantized_recon_t, recon_embeddings_hi), axis=1)
                # recon_image_recon = model_decode(quant_join)

                # # Generate the anomaly segmentation map
                # out_mask = decoder_seg(detach_temp(recon_image_recon) ,detach_temp(recon_image_general))    # question
                # # out_mask_sm = mindspore.softmax(out_mask, dim=1) question:在minds里面好像没有直接的softmax操作
                # temp_softmax = mindspore.nn.softmax(axis=1)
                # out_mask_sm = temp_softmax(out_mask)

                # # Calculate losses
                # loss_feat_hi = mindspore.ops.mse_loss(recon_feat_hi, detach_temp(quantized_b_real)) # question
                # loss_feat_lo = mindspore.ops.mse_loss(recon_feat_lo, detach_temp(quantized_t))  # question
                # loss_l2_recon_img = mindspore.ops.mse_loss(in_image, recon_image_recon)
                # total_recon_loss = loss_feat_lo + loss_feat_hi + loss_l2_recon_img*10


                # # Resize the ground truth anomaly map to closely match the augmented features
                # down_ratio_x_hi = int(anomaly_mask.shape[3] / quantized_b.shape[3])
                # anomaly_mask_hi = mindspore.ops.adaptive_max_pool2d(anomaly_mask,   # question: 这里本应该是torch里面的max_pool2d操作
                #                                                   (down_ratio_x_hi, down_ratio_x_hi)).float()
                # anomaly_mask_hi = mindspore.ops.interpolate(anomaly_mask_hi, scale_factor=down_ratio_x_hi*1.0)  # 这里好像必须是浮点数
                # down_ratio_x_lo = int(anomaly_mask.shape[3] / quantized_t.shape[3])
                # anomaly_mask_lo = mindspore.ops.adaptive_max_pool2d(anomaly_mask,
                #                                                   (down_ratio_x_lo, down_ratio_x_lo)).float()
                # anomaly_mask_lo = mindspore.ops.interpolate(anomaly_mask_lo, scale_factor=down_ratio_x_lo*1.0)
                # anomaly_mask = anomaly_mask_lo * use_both + (
                #             anomaly_mask_lo * use_lo + anomaly_mask_hi * use_hi) * (1.0 - use_both)


                # # Calculate the segmentation loss with GT mask generated at low resolution.
                # segment_loss = loss_focal(out_mask_sm, anomaly_mask)

                # loss = segment_loss + total_recon_loss
                # # loss.backward()
                # # optimizer.step()

                # if i_batch == 0:
                #     print("Loss Focal: ", segment_loss.item())
                #     print("Loss Recon: ", total_recon_loss.item())

                # n_iter +=1
                
                print("fuck you")

    #         # scheduler.step() # question

    #         # if (epoch+1) % 5 == 0:
    #         #     # Save models
    #         #     torch.save(decoder_seg.state_dict(), out_path+"checkpoints/"+run_name+"anomaly_det_module_"+obj_name+".pckl")
    #         #     torch.save(sub_res_model_lo.state_dict(), out_path+"checkpoints/"+run_name+"subspace_restriction_lo_"+obj_name+".pckl")
    #         #     torch.save(sub_res_model_hi.state_dict(), out_path+"checkpoints/"+run_name+"subspace_restriction_hi_"+obj_name+".pckl")
    #         #     torch.save(model_decode.state_dict(), out_path+"checkpoints/"+run_name+"image_recon_module_"+obj_name+".pckl")

    return model, sub_res_model_hi, sub_res_model_lo, model_decode, decoder_seg


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--out_path', action='store', type=str, required=True)

    args = parser.parse_args()

    # Use: python train_dsr.py --gpu_id 0 --obj_id $i --lr 0.0002 --bs 8 --epochs 100 --data_path $BASE_PATH --out_path $OUT_PATH
    # BASE_PATH -- the base directory of mvtec
    # OUT_PATH -- where the trained models will be saved
    # i -- the index of the object class in the obj_batch list
    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    # with torch.cuda.device(args.gpu_id):
    model, sub_res_model_hi, sub_res_model_lo, model_decode, decoder_seg = train_on_device(obj_batch[int(args.obj_id)], args.data_path, args.out_path, args.lr, args.bs, args.epochs)
        # train_upsampling_module(model, sub_res_model_hi, sub_res_model_lo, model_decode, decoder_seg, # 这个后面再改
        #                         obj_batch[int(args.obj_id)], args.data_path, args.out_path, args.lr, args.bs, args.epochs)

