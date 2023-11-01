python -B train_scannet.py \
  --device_target GPU \
  --device_id 0 \
  --batch_size 3 \
  --labeled_point 1% \
  --scale \
  --name psd_scannet_1%-gpu \
  --outputs_dir ./runs

python -B eval_scannet.py \
  --model_path runs/psd_scannet_1%-gpu \
  --device_id 0 \
  --device_target GPU \
  --batch_size 15