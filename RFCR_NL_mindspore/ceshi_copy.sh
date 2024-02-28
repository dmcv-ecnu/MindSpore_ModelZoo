cuda_visible=0
device=0
name=rfcr-area5_weak_inter
pretrain_name=rfcr-area5_weak_pretrain
train_batch_size=3
eval_batch_size=20
# CUDA_VISIBLE_DEVICES=$cuda_visible python -B train.py \
#                                             --mode pretrain \
#                                             --val_area Area_5 \
#                                             --labeled_point 1 \
#                                             --name $pretrain_name \
#                                             --outputs_dir ./runs \
#                                             --device_target GPU \
#                                             --batch_size $train_batch_size \
#                                             --epochs 100 \
#                                             --device_id $device \
#                                             --scale
# CUDA_VISIBLE_DEVICES=$cuda_visible python -B weak_pseudo.py \
#                                             --val_area Area_5 \
#                                             --labeled_point 1 \
#                                             --model_path runs/$pretrain_name \
#                                             --device_id $device \
#                                             --batch_size $eval_batch_size \
#                                             --device_target GPU 
# CUDA_VISIBLE_DEVICES=$cuda_visible python -B weak_test.py \
#                                             --val_area Area_5 \
#                                             --labeled_point 1 \
#                                             --model_path runs/$pretrain_name \
#                                             --device_id $device \
#                                             --batch_size $eval_batch_size \
#                                             --device_target GPU 

# CUDA_VISIBLE_DEVICES=$cuda_visible python -B train.py \
#                                             --mode weak_train \
#                                             --val_area Area_5 \
#                                             --labeled_point 1 \
#                                             --name ${name}1 \
#                                             --outputs_dir ./runs \
#                                             --model_path runs/$pretrain_name \
#                                             --gt_label_path runs/$pretrain_name/gt_1 \
#                                             --pseudo_label_path runs/$pretrain_name/prediction/pseudo_label \
#                                             --device_target GPU \
#                                             --epochs 30 \
#                                             --batch_size $train_batch_size \
#                                             --device_id $device \
#                                             --scale
# CUDA_VISIBLE_DEVICES=$cuda_visible python -B weak_pseudo.py \
#                                             --val_area Area_5 \
#                                             --labeled_point 1 \
#                                             --model_path runs/${name}1 \
#                                             --device_id $device \
#                                             --batch_size $eval_batch_size \
#                                             --device_target GPU 
# CUDA_VISIBLE_DEVICES=$cuda_visible python -B weak_test.py \
#                                             --val_area Area_5 \
#                                             --labeled_point 1 \
#                                             --model_path runs/${name}1 \
#                                             --device_id $device \
#                                             --batch_size $eval_batch_size \
#                                             --device_target GPU 

# CUDA_VISIBLE_DEVICES=$cuda_visible python -B train.py \
#                                             --mode weak_train \
#                                             --val_area Area_5 \
#                                             --labeled_point 1 \
#                                             --name ${name}2 \
#                                             --outputs_dir ./runs \
#                                             --model_path runs/${name}1 \
#                                             --gt_label_path runs/$pretrain_name/gt_1 \
#                                             --pseudo_label_path runs/${name}1/prediction/pseudo_label \
#                                             --device_target GPU \
#                                             --epochs 30 \
#                                             --batch_size $train_batch_size \
#                                             --device_id $device \
#                                             --scale
CUDA_VISIBLE_DEVICES=$cuda_visible python -B weak_pseudo.py \
                                            --val_area Area_5 \
                                            --labeled_point 1 \
                                            --model_path runs/${name}2_old \
                                            --device_id $device \
                                            --batch_size $eval_batch_size \
                                            --device_target GPU 
# CUDA_VISIBLE_DEVICES=$cuda_visible python -B weak_test.py \
#                                             --val_area Area_5 \
#                                             --labeled_point 1 \
#                                             --model_path runs/${name}2 \
#                                             --device_id $device \
#                                             --batch_size $eval_batch_size \
#                                             --device_target GPU 

for i in {3..10}
do
  j=$((i-1))
  CUDA_VISIBLE_DEVICES=$cuda_visible python -B train.py \
                                            --mode weak_train \
                                            --val_area Area_5 \
                                            --labeled_point 1 \
                                            --name ${name}${i}_old \
                                            --outputs_dir ./runs \
                                            --model_path runs/${name}${j}_old \
                                            --gt_label_path runs/$pretrain_name/gt_1 \
                                            --pseudo_label_path runs/${name}${j}_old/prediction/pseudo_label \
                                            --device_target GPU \
                                            --epochs 30 \
                                            --batch_size $train_batch_size \
                                            --device_id $device \
                                            --scale
  CUDA_VISIBLE_DEVICES=$cuda_visible python -B weak_pseudo.py \
                                            --val_area Area_5 \
                                            --labeled_point 1 \
                                            --model_path runs/${name}${i}_old \
                                            --device_id $device \
                                            --batch_size $eval_batch_size \
                                            --device_target GPU 
  CUDA_VISIBLE_DEVICES=$cuda_visible python -B weak_test.py \
                                            --val_area Area_5 \
                                            --labeled_point 1 \
                                            --model_path runs/${name}${i}_old \
                                            --device_id $device \
                                            --batch_size $eval_batch_size \
                                            --device_target GPU 
done