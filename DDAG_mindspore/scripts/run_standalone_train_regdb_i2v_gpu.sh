#!/usr/bin/env bash
myfile="run_standalone_train_regdb_i2v_gpu.sh"

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
--gpu 0 \
--device-target GPU \
--pretrain "resnet50.ckpt" \
--tag "regdb_i2v" \
--data-path "Define your own path/regdb" \
--loss-func "id+tri" \
--branch main \
--regdb-mode "i2v" \
--part 0 \
--graph True \
--epoch 80