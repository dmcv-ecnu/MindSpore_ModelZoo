#!/usr/bin/env bash
myfile="run_eval_regdb_v2i_gpu.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/ and run. Exit..."
    exit 0
fi

cd ..

# Note: --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../DDAG_mindspore/)

python eval.py \
--MSmode "GRAPH_MODE" \
--dataset RegDB \
--gpu 0 \
--device-target GPU \
--resume "XXX.ckpt" \
--tag "regdb_v2i" \
--data-path "Define your own path/regdb" \
--branch main \
--regdb-mode "v2i" \
--part 0