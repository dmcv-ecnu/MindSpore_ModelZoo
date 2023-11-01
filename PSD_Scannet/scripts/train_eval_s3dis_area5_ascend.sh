python -B train_mask.py --device_target Ascend --device_id 0 --batch_size 3 --labeled_point 1% --val_area 5 --scale --name psd_Area-5-ascend --outputs_dir ./runs
python -B test.py --model_path runs/psd_Area-5-ascend --device_id 0 --device_target gpu --batch_size 32 --val_area 5
