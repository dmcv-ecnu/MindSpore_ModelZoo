#!/usr/bin/env bash
myfile="run_standalone_train_regdb_v2i_ascend.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/ and run. Exit..."
    exit 0
fi

cd ..

# Note: --pretrain, --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../DDAG_mindspore/)

python train.py \
--MSmode "GRAPH_MODE" \
--dataset RegDB \
--optim adam \
--lr 0.0035 \
--device-id 0 \
--device-target Ascend \
--pretrain "resnet50.ckpt" \
--tag "regdb_v2i" \
--data-path "Define your own path/regdb" \
--loss-func "id+tri" \
--branch main \
--regdb-mode "v2i" \
--part 0 \
--graph True \
--epoch 80