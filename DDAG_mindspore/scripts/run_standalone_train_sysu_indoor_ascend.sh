#!/usr/bin/env bash
myfile="run_standalone_train_sysu_indoor_ascend.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/ and run. Exit..."
    exit 0
fi

cd ..

# Note: --pretrain, --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../DDAG_mindspore/)

python train.py \
--MSmode "GRAPH_MODE" \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--device-id 0 \
--device-target Ascend \
--pretrain "resnet50.ckpt" \
--tag "sysu_indoor_part_graph" \
--data-path "Define your own path/sysu" \
--loss-func "id+tri" \
--branch main \
--sysu-mode "indoor" \
--part 3 \
--graph True \
--epoch 30