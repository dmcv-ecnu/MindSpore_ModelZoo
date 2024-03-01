CUDA_VISIBLE_DEVICES=2 python -B main_S3DIS_weak_pretrain.py --mode train --test_area 5 --labeled_point 1 --log_dir rfcr_Area-6-pretrain
CUDA_VISIBLE_DEVICES=2 python -B main_S3DIS_weak_pseudo.py --test_area 5 --labeled_point 1 --log_dir rfcr_Area-6-pretrain
CUDA_VISIBLE_DEVICES=2 python -B main_S3DIS_weak_test.py --test_area 5 --labeled_point 1 --log_dir rfcr_Area-6-pretrain
CUDA_VISIBLE_DEVICES=2 python -B main_S3DIS_weak_train.py --test_area 5 --labeled_point 1 --log_dir rfcr_Area-5-iteration1 --load_dir rfcr_Area-6-pretrain --gt_label_path './experiment/S3DIS/1_points_/rfcr_Area-5-pretrain-fix/gt_1' --pseudo_label_path "./experiment/S3DIS/1_points_/rfcr_Area-5-pretrain-fix/prediction/pseudo_label"
CUDA_VISIBLE_DEVICES=2 python -B main_S3DIS_weak_pseudo.py --test_area 5 --labeled_point 1 --log_dir rfcr_Area-5-iteration1
CUDA_VISIBLE_DEVICES=2 python -B main_S3DIS_weak_test.py --test_area 5 --labeled_point 1 --log_dir rfcr_Area-5-iteration1
for i in {2..10}
do
  j=$((i-1))
  CUDA_VISIBLE_DEVICES=2 python -B main_S3DIS_weak_train.py --test_area 5 --labeled_point 1 --log_dir rfcr_Area-5-iteration$i --load_dir rfcr_Area-5-iteration${j} --gt_label_path './experiment/S3DIS/1_points_/rfcr_Area-5-pretrain-fix/gt_1' --pseudo_label_path "./experiment/S3DIS/1_points_/rfcr_Area-5-iteration${j}/prediction/pseudo_label"
  CUDA_VISIBLE_DEVICES=2 python -B main_S3DIS_weak_pseudo.py --test_area 5 --labeled_point 1 --log_dir rfcr_Area-5-iteration$i
  CUDA_VISIBLE_DEVICES=2 python -B main_S3DIS_weak_test.py --test_area 5 --labeled_point 1 --log_dir rfcr_Area-5-iteration$i
done