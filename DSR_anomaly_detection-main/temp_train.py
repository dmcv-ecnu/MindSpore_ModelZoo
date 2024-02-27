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
import mindspore.nn as nn

def detach_temp(original_tensor):
    detached_tensor = mindspore.Tensor(original_tensor)
    detached_tensor.requires_grad = False  
    return detached_tensor

def generate_fake_anomalies_joined(features, embeddings, memory_torch_original, mask, diversity=1.0, strength=None): 
    # (8, 128, 32, 32)  (8, 128, 32, 32)  (4096, 128)  (8, 1, 256, 256)
    random_embeddings = ops.zeros((embeddings.shape[0], embeddings.shape[2] * embeddings.shape[3], memory_torch_original.shape[1])) # (8, 1024, 128)
    inputs = features.permute(0, 2, 3, 1)  # (8 32 32 128)

    for k in range(embeddings.shape[0]): # (0 - 8)
        memory_torch = memory_torch_original
        flat_input = inputs[k].view(-1, memory_torch.shape[1]) # (1024, 128)

        distances_b = (ops.sum(flat_input ** 2, dim=1, keepdim=True)
                     + ops.sum(memory_torch ** 2, dim=1)
                     - 2 * ops.matmul(flat_input, memory_torch.t())) # (1024, 4096)
        
        percentage_vectors = strength[k]
        topk = max(1, min(int(percentage_vectors * memory_torch.shape[0]) + 1, memory_torch.shape[0] - 1))
        values, topk_indices = ops.topk(distances_b, topk, dim=1, largest=False)
        topk_indices = topk_indices[:, int(memory_torch.shape[0] * 0.05):]
        topk = topk_indices.shape[1]

        random_indices_hik = ops.randint(0, topk, size=(topk_indices.shape[0],))
        random_indices_t = topk_indices[ops.arange(random_indices_hik.shape[0]),random_indices_hik]
        random_embeddings[k] = memory_torch[random_indices_t,:]
    
    random_embeddings = random_embeddings.reshape((random_embeddings.shape[0],embeddings.shape[2],embeddings.shape[3],random_embeddings.shape[2])) # (8, 32, 32, 128)
    random_embeddings_tensor = random_embeddings.permute(0,3,1,2) # (8, 128, 32, 32)

    down_ratio_y = int(mask.shape[2]/embeddings.shape[2])
    down_ratio_x = int(mask.shape[3]/embeddings.shape[3]) # 8
    temp = int(mask.shape[3]/down_ratio_x)
    anomaly_mask = mindspore.ops.adaptive_max_pool2d(mask, (temp, temp)).float()    # (8, 1, 8, 8) 这个函数导致shape存在问题

    anomaly_embedding = anomaly_mask * random_embeddings_tensor + (1.0 - anomaly_mask) * embeddings 

    return anomaly_embedding


def train_upsampling_module(model, sub_res_model_hi, sub_res_model_lo, model_decode, decoder_seg, obj_name, mvtec_path, out_path, lr, batch_size, epochs, anom_par):
    run_name = 'dsr_' + str(lr) + '_' + str(epochs) + '_bs' + str(batch_size) + "_anom_par" + str(anom_par) + "_"

    embedding_dim = 128

    model.set_train(False)
    sub_res_model_hi.set_train(False)
    sub_res_model_lo.set_train(False)
    decoder_seg.set_train(False)
    model_decode.set_train(False)

    model_upsample = UpsamplingModule(embedding_size=embedding_dim)
    model_upsample.set_train(True)

    optimizer = mindspore.nn.Adam(model_upsample.trainable_params(), learning_rate=lr)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[int(epochs*0.80), int(epochs*0.90)],gamma=0.2, last_epoch=-1)

    loss_focal = FocalLoss()

    dataset = MVTecImageAnomTrainDataset(mvtec_path + obj_name + "/train/good/", resize_shape=[256, 256])
    dataloader = GeneratorDataset(dataset, shuffle=True, num_parallel_workers=1)
    dataloader = dataloader.batch(batch_size)

    n_iter = 0.0
    segment_loss_avg = 0.0

    for epoch in range(epochs//2):
        start_time = time.time()
        for i_batch, sample_batched in enumerate(dataloader):
            input_image_aug = sample_batched["augmented_image"]
            anomaly_mask = sample_batched["anomaly_mask"]

            loss_b, loss_t, data_recon, embeddings_t, embeddings = model(input_image_aug)

            data_recon = data_recon.detach()
            embeddings = embeddings.detach()
            embeddings_t = embeddings_t.detach()

            embedder = model._vq_vae_bot
            embedder_top = model._vq_vae_top

            anomaly_embedding_copy = embeddings.copy()
            anomaly_embedding_top_copy = embeddings_t.copy()
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

            segment_loss_avg = segment_loss_avg * 0.95 + 0.05 * segment_loss.item()

            n_iter +=1

        # if epoch % 5 == 0:
        #     torch.save(model_upsample.state_dict(), out_path+"checkpoints/"+run_name+"_upsample.pckl")


class DSR_model(nn.Cell):
    def __init__(self, num_hiddens, num_residual_hiddens, num_residual_layers, embedding_dim, num_embeddings, commitment_cost, decay, anom_par):
        super(DSR_model, self).__init__()

        
        model = DiscreteLatentModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay)
        
        self.model = model
        self.embedder_hi = model._vq_vae_bot
        self.embedder_lo = model._vq_vae_top
        
        self.sub_res_model_lo = SubspaceRestrictionModule(embedding_size=embedding_dim)
        self.sub_res_model_hi = SubspaceRestrictionModule(embedding_size=embedding_dim)
        self.decoder_seg = AnomalyDetectionModule(embedding_size=embedding_dim)
        self.model_decode = ImageReconstructionNetwork(embedding_dim * 2, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.loss_focal = FocalLoss()
        self.anom_par = anom_par
        
    def construct(self, in_image, anomaly_mask):
        anomaly_strength_lo = (ops.rand(in_image.shape[0]) * (1.0-self.anom_par) + self.anom_par)
        anomaly_strength_hi = (ops.rand(in_image.shape[0]) * (1.0-self.anom_par) + self.anom_par)
        
        # Extract features from the discrete model
        enc_b = self.model._encoder_b(in_image)
        enc_t = self.model._encoder_t(enc_b)
        zt = self.model._pre_vq_conv_top(enc_t)
        
        # Quantize the extracted features
        loss_t, quantized_t, perplexity_t, encodings_t = self.embedder_lo(zt)  # quantized_t.shape 8 128 32 3
            
        # Generate feature-based anomalies on F_lo
        anomaly_embedding_lo = generate_fake_anomalies_joined(zt, quantized_t,
                                                                self.embedder_lo._embedding.embedding_table,
                                                                anomaly_mask, strength=anomaly_strength_lo)
        
        # Upsample the extracted quantized features and the quantized features augmented with anomalies
        up_quantized_t = self.model.upsample_t(anomaly_embedding_lo)     # may here
        up_quantized_t_real = self.model.upsample_t(quantized_t)

        feat = ops.cat((enc_b, up_quantized_t), axis=1)
        feat_real = ops.cat((enc_b, up_quantized_t_real), axis=1)
        zb = self.model._pre_vq_conv_bot(feat)
        zb_real = self.model._pre_vq_conv_bot(feat_real)
        
        # Quantize the upsampled features - F_hi
        loss_b, quantized_b, perplexity_b, encodings_b = self.embedder_hi(zb)
        loss_b, quantized_b_real, perplexity_b, encodings_b = self.embedder_hi(zb_real)

        # Generate feature-based anomalies on F_hi
        anomaly_embedding = generate_fake_anomalies_joined(zb, quantized_b,
                                                                self.embedder_hi._embedding.embedding_table, anomaly_mask
                                                                , strength=anomaly_strength_hi)
        
        use_both = ops.randint(0, 2,(in_image.shape[0],1,1,1)).float()
        use_lo = ops.randint(0, 2,(in_image.shape[0],1,1,1)).float()
        use_hi = (1 - use_lo)
        anomaly_embedding_hi_usebot = generate_fake_anomalies_joined(zb_real,
                                                                quantized_b_real,
                                                                self.embedder_hi._embedding.embedding_table,
                                                            anomaly_mask, strength=anomaly_strength_hi)

        anomaly_embedding_lo_usebot = quantized_t
        anomaly_embedding_hi_usetop = quantized_b_real
        anomaly_embedding_lo_usetop = anomaly_embedding_lo
        anomaly_embedding_hi_not_both =  use_hi * anomaly_embedding_hi_usebot + use_lo * anomaly_embedding_hi_usetop
        anomaly_embedding_lo_not_both =  use_hi * anomaly_embedding_lo_usebot + use_lo * anomaly_embedding_lo_usetop
        # anomaly_embedding_hi = (anomaly_embedding * use_both + anomaly_embedding_hi_not_both * (1.0 - use_both)).detach().clone()
        # anomaly_embedding_lo = (anomaly_embedding_lo * use_both + anomaly_embedding_lo_not_both * (1.0 - use_both)).detach().clone()
        anomaly_embedding_hi = detach_temp(anomaly_embedding * use_both + anomaly_embedding_hi_not_both * (1.0 - use_both)).copy()
        anomaly_embedding_lo = detach_temp(anomaly_embedding_lo * use_both + anomaly_embedding_lo_not_both * (1.0 - use_both)).copy()

        anomaly_embedding_hi_copy = anomaly_embedding_hi.copy()
        anomaly_embedding_lo_copy = anomaly_embedding_lo.copy()   # question

        # Restore the features to normality with the Subspace restriction modules
        recon_feat_hi, recon_embeddings_hi, loss_b = self.sub_res_model_hi(anomaly_embedding_hi_copy, self.embedder_hi)
        recon_feat_lo, recon_embeddings_lo, loss_b_lo = self.sub_res_model_lo(anomaly_embedding_lo_copy, self.embedder_lo)

        # Reconstruct the image from the anomalous features with the general appearance decoder
        up_quantized_anomaly_t = self.model.upsample_t(anomaly_embedding_lo)
        quant_join_anomaly = ops.cat((up_quantized_anomaly_t, anomaly_embedding_hi), axis=1)
        recon_image_general = self.model._decoder_b(quant_join_anomaly)


        # Reconstruct the image from the reconstructed features
        # with the object-specific image reconstruction module
        up_quantized_recon_t = self.model.upsample_t(recon_embeddings_lo)
        quant_join = ops.cat((up_quantized_recon_t, recon_embeddings_hi), axis=1)
        recon_image_recon = self.model_decode(quant_join)

        # Generate the anomaly segmentation map
        out_mask = self.decoder_seg(detach_temp(recon_image_recon) ,detach_temp(recon_image_general))    # question
        out_mask_sm = mindspore.ops.softmax(out_mask, axis=1)

        # Calculate losses
        loss_feat_hi = mindspore.ops.mse_loss(recon_feat_hi, detach_temp(quantized_b_real)) # question
        loss_feat_lo = mindspore.ops.mse_loss(recon_feat_lo, detach_temp(quantized_t))  # question
        loss_l2_recon_img = mindspore.ops.mse_loss(in_image, recon_image_recon)
        total_recon_loss = loss_feat_lo + loss_feat_hi + loss_l2_recon_img*10


        # Resize the ground truth anomaly map to closely match the augmented features
        down_ratio_x_hi = int(anomaly_mask.shape[3] / quantized_b.shape[3])
        temp_1 = int(anomaly_mask.shape[3] / down_ratio_x_hi)
        anomaly_mask_hi = mindspore.ops.adaptive_max_pool2d(anomaly_mask,   # question: 这里本应该是torch里面的max_pool2d操作
                                                            (temp_1, temp_1)).float()
        anomaly_mask_hi = mindspore.ops.interpolate(anomaly_mask_hi, mode="area", scale_factor=down_ratio_x_hi*1.0)  # 这里好像必须是浮点数
        down_ratio_x_lo = int(anomaly_mask.shape[3] / quantized_t.shape[3])
        temp_2 = int(anomaly_mask.shape[3] / down_ratio_x_lo)
        anomaly_mask_lo = mindspore.ops.adaptive_max_pool2d(anomaly_mask,
                                                            (temp_2, temp_2)).float()
        anomaly_mask_lo = mindspore.ops.interpolate(anomaly_mask_lo, mode="area", scale_factor=down_ratio_x_lo*1.0)
        
        anomaly_mask = anomaly_mask_lo * use_both + (
                    anomaly_mask_lo * use_lo + anomaly_mask_hi * use_hi) * (1.0 - use_both)

        # Calculate the segmentation loss with GT mask generated at low resolution.
        segment_loss = self.loss_focal(out_mask_sm, anomaly_mask)
        loss = segment_loss + total_recon_loss
        return loss


def train_on_device(obj_names, mvtec_path, out_path, lr, batch_size, epochs):
    run_name_pre = 'vq_model_pretrained_128_4096'   # 需要预训练模块
    context.set_context(device_target="GPU", device_id=1)

    for obj_name in obj_names:
        Dsr_model = DSR_model(num_hiddens = 128, num_residual_hiddens = 64, num_residual_layers = 2, embedding_dim = 128, num_embeddings = 4096, commitment_cost = 0.25, decay = 0.99, anom_par = 0.2)
        group_params = [{'params': Dsr_model.sub_res_model_lo.trainable_params(), 'lr': lr},     
                        {'params': Dsr_model.sub_res_model_hi.trainable_params(), 'lr': lr},
                        {'params': Dsr_model.model_decode.trainable_params(), 'lr': lr},
                        {'params': Dsr_model.decoder_seg.trainable_params(), 'lr': lr}]
        optimizer = mindspore.nn.Adam(group_params, learning_rate=0.001, weight_decay=0.0, use_lazy=False, use_offload=False)
        
        # Define the dataset and dataloader
        dataset = TrainWholeImageDataset(mvtec_path + obj_name + "/train/good/", resize_shape=[256, 256], perlin_augment=True)
        dataloader = mindspore.dataset.GeneratorDataset(dataset, column_names=["image", "mask", "is_normal", "idx"], num_parallel_workers=1, shuffle=True)   
        print(len(dataloader))
        dataloader = dataloader.batch(batch_size=batch_size)
        print(len(dataloader))
        # Define forward function
        def forward_fn(in_image, anomaly_mask):
            loss = Dsr_model(in_image, anomaly_mask)     
            return loss
            
        # Get gradient function
        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
        
        # Define function of one-step training
        def train_step(in_image, anomaly_mask):
            (loss), grads = grad_fn(in_image, anomaly_mask)
            # loss = ops.depend(loss, optimizer(grads))
            optimizer(grads)
            return loss

        # The training process
        for epoch in range(2):
            print("Epoch ", epoch)
            for i_batch, sample_batched in enumerate(dataloader):
                in_image, anomaly_mask, is_noraml, idx = sample_batched
                loss = train_step(in_image, anomaly_mask)
                loss = loss.asnumpy()
                print("epoch: {}   batch: {}   loss: {}   ".format(epoch, i_batch, loss))

    return Dsr_model.model, Dsr_model.sub_res_model_hi, Dsr_model.sub_res_model_lo, Dsr_model.model_decode, Dsr_model,decoder_seg


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

