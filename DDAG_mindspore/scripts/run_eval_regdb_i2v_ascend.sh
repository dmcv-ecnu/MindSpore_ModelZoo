#!/usr/bin/env bash
myfile="run_eval_regdb_i2v_ascend.sh"

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
--device-id 0 \
--device-target Ascend \
--resume "XXX.ckpt" \
--tag "regdb_i2v" \
--data-path "Define your own path/regdb" \
--branch main \
--regdb-mode "i2v" \
--part 0