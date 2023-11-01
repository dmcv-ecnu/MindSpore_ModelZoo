python -B train_mask.py --device_target GPU --device_id 0 --batch_size 3 --labeled_point 1% --val_area 5 --scale --name psd_Area-5-gpu --outputs_dir ./runs
python -B eval.py --model_path runs/psd_Area-5-gpu --device_id 0 --device_target GPU --batch_size 32 --val_area 5
