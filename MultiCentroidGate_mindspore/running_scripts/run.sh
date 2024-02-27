num_gpus="$(awk -F/ '{print NF+1}' <<< "$CUDA_VISIBLE_DEVICES")"
echo $num_gpus
template='
torchrun --nproc_per_node=1 --master_port $((10000 + $RANDOM % 11451)) src/main.py  --config-files 
        "./conf/config_ok.yaml"
'

eval $template "./conf/cifarbase0step10.yaml" --exp-name "der_b0step10"  \
         --subtrainer "baseline" --part-enable 1 1 0 0 \
         --batch-size 128 --ft.batch-size 128 --expert-type "en"

eval $template "./conf/cifarbase0step10.yaml" --exp-name "mcm_b0step10"  \
         --subtrainer "logit_map" \
         --batch-size 512 --ft.batch-size 512 --expert-type "en" \
         --pretrain-model-dir "$YOUR_SAVE_PATH_OF_PREV_MODEL" \
         --trainer "metric_re" --network "gate_mcm_norm" --save-model --task-temperature 12
 

python src/main.py --config-files "./conf/config_ok.yaml" "./conf/cifarbase0step10.yaml" --exp-name "der_b0step10" --subtrainer "baseline" --part-enable 1 1 0 0 --batch-size 128 --ft.batch-size 128 --expert-type "en"
# torchrun --nproc_per_node=1 --master_port $((10000 + $RANDOM % 11451)) src/main.py --config-files "./conf/config_ok.yaml" "./conf/cifarbase0step10.yaml" --exp-name "der_b0step10" --subtrainer "baseline" --part-enable 1 1 0 0 --batch-size 128 --ft.batch-size 128 --expert-type "en"

# torchrun --nproc_per_node=1 --master_port $((10000 + $RANDOM % 11451)) src/main.py --config-files "./conf/config_ok.yaml" "./conf/cifarbase0step10.yaml" --exp-name "mcm_b0step10" --subtrainer "logit_map" --batch-size 512 --ft.batch-size 512 --expert-type "en" --pretrain-model-dir "path_to_previous_model" --trainer "metric_re" --network "gate_mcm_norm" --save-model --task-temperature 12
